"""T3: disk-persisted hybrid retrieval index (BM25 + FAISS dense + RRF fusion).

- Documents are chunked, embedded with the shared SBERT embedder (the same
  sentence-transformers model the API already loads), and stored both in a
  rank_bm25 corpus and a FAISS index.
- Retrieval fuses BM25 and dense rankings with reciprocal rank fusion (RRF);
  an optional cross-encoder reranker reorders the fused top-k.
- The index persists under data/processed/retrieval_index/ (gitignored) and
  reloads on API startup.
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

INDEX_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval_index"

RRF_K = 60
DEFAULT_TOP_K = 5
RELEVANCE_FLOOR = 1e-6  # below this fused score -> abstain (no evidence)


class RetrievalIndex:
    def __init__(self, embed_fn, index_dir: Path = INDEX_DIR_DEFAULT, top_k: int = DEFAULT_TOP_K):
        """embed_fn(texts: list[str]) -> np.ndarray (n, dim)."""
        self._embed = embed_fn
        self._dir = index_dir
        self._top_k = top_k
        self._lock = threading.Lock()
        self._passages: list[dict] = []          # {id, source, url, text, sha256}
        self._meta: dict = {"documents": [], "dim": None}
        self._faiss = None
        self._bm25 = None
        self._corpus_tokens: list[list] = []
        self.load()

    # ------------------------------------------------------------------ paths
    @property
    def passages_path(self) -> Path:
        return self._dir / "passages.json"

    @property
    def meta_path(self) -> Path:
        return self._dir / "meta.json"

    @property
    def index_path(self) -> Path:
        return self._dir / "vectors.faiss"

    # ------------------------------------------------------------------ state
    def _tokenize(self, text: str) -> list:
        return text.lower().split()

    def load(self) -> None:
        with self._lock:
            if not self.passages_path.exists():
                return
            try:
                self._passages = json.loads(self.passages_path.read_text(encoding="utf-8"))
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if self._passages:
                    self._corpus_tokens = [self._tokenize(p["text"]) for p in self._passages]
                    self._bm25 = self._make_bm25(self._corpus_tokens)
            except (OSError, ValueError):
                self._passages = []
                self._meta = {"documents": [], "dim": None}

    def _make_bm25(self, corpus: list):
        from rank_bm25 import BM25Okapi

        return BM25Okapi(corpus)

    def _build_faiss(self, dim: int) -> None:
        import faiss

        self._faiss = faiss.IndexFlatIP(dim)

    def add_documents(self, documents: list[dict], embed_fn=None) -> dict:
        """documents: [{source, url?, text}]. Returns per-doc chunk counts."""
        embed = embed_fn or self._embed
        new_passages: list[dict] = []
        new_vecs: list[np.ndarray] = []
        result = []
        with self._lock:
            existing = {p["sha256"] for p in self._passages}
            for doc in documents:
                source = doc["source"]
                url = doc.get("url") or ""
                chunks = [c for c in self._chunks_of(doc["text"]) if c.strip()]
                chunk_vecs = embed(chunks)
                n_added = 0
                for i, (chunk, vec) in enumerate(zip(chunks, chunk_vecs)):
                    sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()  # content-based dedup
                    if sha in existing:
                        continue
                    existing.add(sha)
                    new_passages.append({"id": sha, "source": source, "url": url,
                                         "text": chunk, "sha256": sha})
                    new_vecs.append(np.asarray(vec, dtype="float32"))
                    n_added += 1
                result.append({"source": source, "chunks": len(chunks), "added": n_added})
            if new_passages:
                self._passages.extend(new_passages)
                vecs = np.stack(new_vecs).astype("float32")
                dim = vecs.shape[1]
                self._meta["dim"] = dim
                if self._faiss is None or self._faiss.d != dim:
                    self._build_faiss(dim)
                self._faiss.add(vecs)
                self._corpus_tokens.extend(self._tokenize(p["text"]) for p in new_passages)
                self._bm25 = self._make_bm25(self._corpus_tokens)
                self._save()
            return result

    def _chunks_of(self, text: str) -> list:
        from .chunk import chunk_text

        return chunk_text(text)

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self.passages_path.write_text(json.dumps(self._passages, ensure_ascii=False), encoding="utf-8")
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")
        if self._faiss is not None:
            import faiss

            faiss.write_index(self._faiss, str(self.index_path))

    def clear(self) -> None:
        with self._lock:
            self._passages = []
            self._meta = {"documents": [], "dim": None}
            self._faiss = None
            self._bm25 = None
            self._corpus_tokens = []
            self._dir.mkdir(parents=True, exist_ok=True)
            for p in (self.passages_path, self.meta_path, self.index_path):
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------------ stats
    def status(self) -> dict:
        with self._lock:
            return {
                "n_passages": len(self._passages),
                "n_documents": len(self._meta.get("documents", [])),
                "dim": self._meta.get("dim"),
                "index_dir": str(self._dir),
            }

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int | None = None, rerank_fn=None) -> list:
        """Hybrid BM25 + dense retrieval with RRF fusion, optional rerank."""
        k = top_k or self._top_k
        with self._lock:
            if not self._passages:
                return []
            q_tokens = self._tokenize(query)

            dense_scores: np.ndarray | None = None
            if self._faiss is not None:
                qv = np.asarray(self._embed([query])[0], dtype="float32").reshape(1, -1)
                scores, idxs = self._faiss.search(qv, min(k * 3, len(self._passages)))
                dense_scores = np.zeros(len(self._passages))
                for s, i in zip(scores[0], idxs[0]):
                    if i >= 0:
                        dense_scores[i] = s

            bm25_scores = np.asarray(self._bm25.get_scores(q_tokens)) if self._bm25 else None

            def rrf(scores, ascending=False):
                order = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
                out = np.zeros(len(scores))
                for rank, i in enumerate(order):
                    out[i] = 1.0 / (RRF_K + rank + 1)
                return out

            fused = np.zeros(len(self._passages))
            if dense_scores is not None:
                fused += rrf(dense_scores)
            if bm25_scores is not None:
                fused += rrf(bm25_scores)
            if dense_scores is None and bm25_scores is None:
                return []

            order = np.argsort(fused)[::-1]
            candidates = [
                {**self._passages[i], "score": float(fused[i])}
                for i in order[: k * 2]
                if fused[i] > RELEVANCE_FLOOR
            ]

        if candidates and rerank_fn is not None:
            candidates = rerank_fn(query, candidates, k)

        return candidates[:k]
