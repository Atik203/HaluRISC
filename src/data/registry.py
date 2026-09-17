"""
B1 dataset registry: license, provenance, grouping rules, and label rules
(roadmap §14 B1.6/B1.7).

Restricted datasets (FaithBench, CC BY-NC-SA) are NEVER bundled in the
repository: raw files stay under gitignored `data/raw/`, and only download
instructions, citations, hashes, and license notes are shipped.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    display_name: str
    url: str
    license: str
    redistribution_allowed: bool
    grouping_rule: str
    label_definition: str
    label_mapping_version: str
    citation: str
    raw_dir: str  # relative to repo root, under data/raw/


DATASET_REGISTRY = (
    DatasetRecord(
        name="halueval",
        display_name="HaluEval (QA)",
        url="https://github.com/RUCAIBox/HaluEval",
        license="MIT",
        redistribution_allowed=True,
        grouping_rule="group by item_idx (original question); both answer variants stay in one partition",
        label_definition="correct answer -> 0; hallucinated answer -> 1",
        label_mapping_version="b1-labels-v1",
        citation="Li et al., ACL 2023, DOI 10.18653/v1/2023.emnlp-main.397",
        raw_dir="data/raw/halueval",
    ),
    DatasetRecord(
        name="ragtruth",
        display_name="RAGTruth (official)",
        url="https://github.com/ParticleMedia/RAGTruth",
        license="MIT",
        redistribution_allowed=True,
        grouping_rule="group by source_id (one source elicits six responses)",
        label_definition="any human-annotated hallucination span -> 1; no spans -> 0",
        label_mapping_version="b1-labels-v1",
        citation="Niu et al., ACL 2024, DOI 10.18653/v1/2024.acl-long.585",
        raw_dir="data/raw/ragtruth_official",
    ),
    DatasetRecord(
        name="faithbench",
        display_name="FaithBench (summarization)",
        url="https://github.com/vectara/FaithBench",
        license="CC BY-NC-SA 4.0",
        redistribution_allowed=False,
        grouping_rule="group by raw_sample_id when available, else stable hash of source text",
        label_definition="worst-severity aggregation; Benign/empty -> 0; Questionable/Unwanted* -> 1",
        label_mapping_version="b1-labels-v1",
        citation="Bao et al., NAACL 2025, DOI 10.18653/v1/2025.naacl-short.38",
        raw_dir="data/raw/faithbench",
    ),
)


def read_hashes_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def license_manifest() -> dict:
    """Assemble the B1 license/provenance manifest from the registry + disk hashes."""
    datasets = []
    for rec in DATASET_REGISTRY:
        revision = read_hashes_json(ROOT / rec.raw_dir / "revision.json")
        files = {}
        for name, meta in (revision.get("files") or {}).items():
            files[name] = {
                "sha256": meta.get("sha256"),
                "bytes": meta.get("bytes"),
            }
        datasets.append(
            {
                **asdict(rec),
                "download_revision": revision.get("commit_sha"),
                "downloaded_at_utc": revision.get("fetched_at_utc"),
                "files": files,
            }
        )
    return {
        "schema": "b1-license-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Restricted datasets are not redistributed; download instructions, hashes, and license notes are shipped instead.",
        "datasets": datasets,
    }
