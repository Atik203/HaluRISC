"""
Data acquisition script for HaluRISC.

Downloads the HaluEval QA subset (qa_data.json, JSONL) from the official
RUCAIBox/HaluEval repository. Per download it records the repo commit revision
and the file SHA-256 hash in data/raw/halueval/revision.json, mirroring
src/data/download_ragtruth.py and src/data/download_faithbench.py.

Run (repo root, .venv):
  python src/data/download.py
"""

import hashlib
import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_halueval")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "halueval"

REPO = "RUCAIBox/HaluEval"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{REPO}"

# name -> path inside the repo
FILES = {
    "qa_data.json": "data/qa_data.json",
}


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "halurisc-b1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sha256(path: Path) -> str:
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


def _validate_qa_jsonl() -> int:
    """Verify that every non-empty line of qa_data.json parses as JSON."""
    count = 0
    with open(OUT_DIR / "qa_data.json", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    logger.info(f"Validated HaluEval QA: {count} JSONL items loaded successfully.")
    return count


def download_halueval_qa(force: bool = False) -> str:
    """Download HaluEval qa_data.json if missing; returns the file path (str)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    dest = OUT_DIR / "qa_data.json"
    revision_path = OUT_DIR / "revision.json"
    missing = not dest.exists()

    if missing or force:
        if force and not missing:
            logger.info("force=True: re-downloading HaluEval qa_data.json")
        for name, repo_path in FILES.items():
            url = f"{RAW_BASE}/{repo_path}"
            logger.info(f"  {url}")
            urllib.request.urlretrieve(url, OUT_DIR / name)
            size_mb = (OUT_DIR / name).stat().st_size / (1024 * 1024)
            logger.info(f"  saved {name} ({size_mb:.2f} MB)")
    else:
        logger.info("HaluEval qa_data.json already present, skipping download.")

    n_items = _validate_qa_jsonl()

    if missing or force or not revision_path.exists():
        revision_path.write_text(
            json.dumps(
                {
                    "repo": REPO,
                    "branch": BRANCH,
                    "commit_sha": _fetch_commit_sha(),
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "files": {
                        name: {
                            "sha256": _sha256(OUT_DIR / name),
                            "bytes": (OUT_DIR / name).stat().st_size,
                        }
                        for name in FILES
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Revision + hashes written to {revision_path}")

    return str(dest)


if __name__ == "__main__":
    download_halueval_qa()
