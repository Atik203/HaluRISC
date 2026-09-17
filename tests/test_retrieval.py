"""T3 retrieval tests: chunking, hybrid index, web search (mocked), rerank fallback."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.chunk import chunk_text, extract_text_from_bytes  # noqa: E402
from src.retrieval.index import RetrievalIndex  # noqa: E402
from src.retrieval.web_search import WebSearch  # noqa: E402


# ---------------------------------------------------------------- embed stub
class FakeEmbedder:
    def __init__(self, vocab: dict[str, np.ndarray]):
        self._vocab = vocab

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vocab.get(t, np.zeros(8, dtype="float32")) for t in texts])


def _idx(tmp_path, texts: list[str]) -> RetrievalIndex:
    # bag-of-words embedder: dense similarity follows lexical overlap, so
    # hybrid retrieval behaves deterministically in tests
    vocab = sorted({w for t in texts for w in t.lower().split()})
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    def embed(texts_):
        out = []
        for t in texts_:
            v = np.zeros(len(vocab), dtype="float32")
            for w in t.lower().split():
                if w in vocab_idx:
                    v[vocab_idx[w]] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out, dtype="float32")

    index = RetrievalIndex(embed, index_dir=tmp_path / "idx")
    index.add_documents([{"source": f"doc{i}", "text": t} for i, t in enumerate(texts)])
    return index


# ---------------------------------------------------------------- chunking
def test_chunk_small_text_stays_one():
    assert chunk_text("Short text.") == ["Short text."]


def test_chunk_long_text_splits_with_overlap():
    text = " ".join([f"Sentence number {i} has enough words to count." for i in range(60)])
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 1550 for c in chunks)


def test_chunk_empty():
    assert chunk_text("   ") == []


def test_extract_txt_bytes():
    assert "hello" in extract_text_from_bytes("notes.txt", b"hello world")


def test_extract_pdf_requires_pypdf():
    with pytest.raises(Exception):  # pypdf raises PdfStreamError on garbage input
        extract_text_from_bytes("x.pdf", b"not a pdf")


# ---------------------------------------------------------------- hybrid index
def test_index_add_status_and_persist(tmp_path):
    idx = _idx(tmp_path, ["Paris is the capital of France.", "The city has 2 million people."])
    assert idx.status()["n_passages"] == 2
    assert (tmp_path / "idx" / "passages.json").exists()

    idx2 = RetrievalIndex(idx._embed, index_dir=tmp_path / "idx")
    assert idx2.status()["n_passages"] == 2  # reloaded from disk


def test_index_clear(tmp_path):
    idx = _idx(tmp_path, ["Some text."])
    idx.clear()
    assert idx.status()["n_passages"] == 0
    assert not (tmp_path / "idx" / "passages.json").exists()


def test_index_search_returns_ranked_passages(tmp_path):
    texts = [
        "Paris is the capital of France.",
        "The Eiffel Tower stands in Paris.",
        "The capital city of France is Paris, on the Seine.",
        "Bananas are yellow fruit grown in the tropics.",
    ]
    idx = _idx(tmp_path, texts)
    hits = idx.search("capital of France", top_k=2)
    assert len(hits) == 2
    assert "Paris" in hits[0]["text"]
    assert "source" in hits[0]


def test_index_search_empty_corpus(tmp_path):
    idx = RetrievalIndex(lambda t: np.zeros((len(t), 8), dtype="float32"), index_dir=tmp_path / "idx")
    assert idx.search("anything") == []


def test_index_add_dedupes_exact_chunks(tmp_path):
    idx = _idx(tmp_path, ["Same sentence repeated."])
    n_before = idx.status()["n_passages"]
    idx.add_documents([{"source": "docX", "text": "Same sentence repeated."}])
    assert idx.status()["n_passages"] == n_before  # identical chunk not re-added


# ---------------------------------------------------------------- web search
def test_web_search_disabled_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    ws = WebSearch(api_key="")
    assert ws.enabled is False
    assert ws.search("anything") == []


def test_web_search_mocked(monkeypatch):
    class FakeTavily:
        def search(self, query, max_results, search_depth):
            return {"results": [{"url": "https://example.com", "content": "Paris facts", "score": 0.9}]}

    ws = WebSearch(api_key="fake")
    monkeypatch.setattr(ws, "_tavily", lambda: FakeTavily())
    hits = ws.search("capital of France")
    assert hits and hits[0]["source"] == "web:https://example.com"
    assert hits[0]["url"] == "https://example.com"


def test_web_search_caches_per_query(monkeypatch):
    calls = {"n": 0}

    class FakeTavily:
        def search(self, query, max_results, search_depth):
            calls["n"] += 1
            return {"results": [{"url": "https://e.com", "content": "c", "score": 0.5}]}

    ws = WebSearch(api_key="fake")
    monkeypatch.setattr(ws, "_tavily", lambda: FakeTavily())
    assert ws.search("q") == ws.search("q")
    assert calls["n"] == 1


# ---------------------------------------------------------------- rerank
def test_rerank_falls_back_when_model_unavailable(monkeypatch):
    from src.retrieval import rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "_load", lambda: (_ for _ in ()).throw(ImportError("offline")))
    candidates = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    out = rerank_mod.rerank("q", candidates, top_k=2)
    assert [c["id"] for c in out] == ["a", "b"]  # unchanged order, no crash
