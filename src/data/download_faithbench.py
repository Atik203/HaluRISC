"""
FaithBench (NAACL 2025) acquisition for HaluRISC Version B (B1).

Downloads the official human-annotated release batches from vectara/FaithBench
(data_for_release/batch_{1..16}.json; batch 13 does not exist upstream).

FaithBench is CC BY-NC-SA 4.0: the raw files stay under the gitignored
`data/raw/faithbench/` and are NEVER bundled in the repository. Only
download instructions, citations, hashes, and license notes are shipped.

Run (repo root, .venv):
  python src/data/download_faithbench.py
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_faithbench")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "faithbench"

REPO = "vectara/FaithBench"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{REPO}"

# batch_13 does not exist in the official release
BATCH_IDS = [i for i in range(1, 17) if i != 13]
BATCH_FILE = "batch_{id}.json"


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
    except Exception as e:
        logger.warning(f"Could not fetch commit revision: {e}")
        return "unknown"


def _validate_batches() -> int:
    """Each batch parses and every sample has source/summary/metadata."""
    total = 0
    for batch_id in BATCH_IDS:
        path = OUT_DIR / BATCH_FILE.format(id=batch_id)
        batch = json.loads(path.read_text(encoding="utf-8"))
        samples = batch["samples"]
        for s in samples:
            assert isinstance(s.get("source"), str) and s["source"].strip()
            assert isinstance(s.get("summary"), str) and s["summary"].strip()
            assert "metadata" in s and "summarizer" in s["metadata"]
        total += len(samples)
    logger.info(f"Validated {len(BATCH_IDS)} FaithBench batches, {total} samples total.")
    return total


def download_faithbench(force: bool = False) -> dict:
    """Download the official batches if missing; returns {batch_id: path}."""
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {}
    missing = [b for b in BATCH_IDS if not (OUT_DIR / BATCH_FILE.format(id=b)).exists()]

    if missing or force:
        if force and not missing:
            logger.info("force=True: re-downloading FaithBench batches")
        commit_sha = _fetch_commit_sha()
        for batch_id in BATCH_IDS:
            name = BATCH_FILE.format(id=batch_id)
            dest = OUT_DIR / name
            if dest.exists() and not force:
                paths[batch_id] = str(dest)
                continue
            url = f"{RAW_BASE}/data_for_release/{name}"
            logger.info(f"  {url}")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"  saved {name} ({dest.stat().st_size / 1024:.0f} KB)")
        _validate_batches()
        hashes = {
            name: {"sha256": _sha256(OUT_DIR / name), "bytes": (OUT_DIR / name).stat().st_size}
            for name in sorted(p.name for p in OUT_DIR.glob("batch_*.json"))
        }
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
        logger.info("FaithBench batches already present, skipping.")

    for batch_id in BATCH_IDS:
        paths[batch_id] = str(OUT_DIR / BATCH_FILE.format(id=batch_id))
    return paths


def load_faithbench_samples():
    """Returns a list of (batch_id, sample_dict) across all official batches."""
    paths = download_faithbench()
    samples = []
    for batch_id, path in paths.items():
        batch = json.loads(Path(path).read_text(encoding="utf-8"))
        for s in batch["samples"]:
            samples.append((int(batch_id), s))
    return samples


if __name__ == "__main__":
    download_faithbench()
