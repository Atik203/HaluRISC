"""
RAGTruth (ACL 2024) OFFICIAL acquisition for HaluRISC Version B (B1).

Downloads the two official files from ParticleMedia/RAGTruth:
  dataset/response.jsonl    (responses with word-level hallucination spans)
  dataset/source_info.jsonl (sources, task types, prompts)

This replaces the lossy HuggingFace mirror used in Version A (which dropped
source_id, spans, task_type, split, and quality). B1 keeps every field.

Per download it records the repo commit revision and per-file SHA-256 hashes
in data/raw/ragtruth_official/revision.json for provenance.

Run (repo root, .venv):
  python src/data/download_ragtruth.py
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_ragtruth")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "ragtruth_official"

REPO = "ParticleMedia/RAGTruth"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{REPO}"

# name -> path inside the repo
FILES = {
    "response.jsonl": "dataset/response.jsonl",
    "source_info.jsonl": "dataset/source_info.jsonl",
}


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "halurisc-b1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_commit_sha() -> str:
    try:
        info = _http_get_json(f"{API_BASE}/commits/{BRANCH}")
        return str(info.get("sha", "unknown"))
    except Exception as e:  # provenance should not block the download
        logger.warning(f"Could not fetch commit revision: {e}")
        return "unknown"


def _validate_official():
    """Structural validation of the official files (B1: reproducible download)."""
    resp_path = OUT_DIR / "response.jsonl"
    src_path = OUT_DIR / "source_info.jsonl"

    responses = []
    source_ids = set()
    with open(resp_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                src = json.loads(line)
                source_ids.add(src["source_id"])

    ids = [r["id"] for r in responses]
    assert len(ids) == len(set(ids)), "response ids are not unique"
    assert len(source_ids) == len({str(s) for s in source_ids}), "source ids are not unique"
    missing = sorted({r["source_id"] for r in responses} - {str(s) for s in source_ids})
    assert not missing, f"responses reference unknown sources: {missing[:5]}"
    logger.info(
        f"Validated official RAGTruth: {len(responses)} responses, "
        f"{len(source_ids)} sources, no orphan responses."
    )
    return len(responses), len(source_ids)


def download_ragtruth_official(force: bool = False) -> dict:
    """Download the official files if missing; returns {name: path}."""
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {}
    missing = [name for name in FILES if not (OUT_DIR / name).exists()]

    if missing or force:
        if force and not missing:
            logger.info("force=True: re-downloading official RAGTruth files")
        commit_sha = _fetch_commit_sha()
        for name, repo_path in FILES.items():
            url = f"{RAW_BASE}/{repo_path}"
            dest = OUT_DIR / name
            logger.info(f"  {url}")
            urllib.request.urlretrieve(url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info(f"  saved {dest.name} ({size_mb:.2f} MB)")
        _validate_official()
        hashes = {name: {"sha256": _sha256(OUT_DIR / name), "bytes": (OUT_DIR / name).stat().st_size} for name in FILES}
        (OUT_DIR / "revision.json").write_text(
            json.dumps(
                {
                    "repo": REPO,
                    "branch": BRANCH,
                    "commit_sha": commit_sha,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "files": hashes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Revision + hashes written to {OUT_DIR / 'revision.json'}")
    else:
        logger.info("Official RAGTruth files already present, skipping.")

    for name in FILES:
        paths[name] = str(OUT_DIR / name)
    return paths


def load_ragtruth_official():
    """Returns (responses: list[dict], sources: dict[source_id -> dict])."""
    paths = download_ragtruth_official()
    responses = []
    with open(paths["response.jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    sources = {}
    with open(paths["source_info.jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                src = json.loads(line)
                sources[str(src["source_id"])] = src
    return responses, sources


if __name__ == "__main__":
    download_ragtruth_official()
