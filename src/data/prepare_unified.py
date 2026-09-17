"""
B1 — Unified dataset builder (roadmap §14 B1).

Maps HaluEval, RAGTruth (official), and FaithBench into the canonical schema
(src/data/schema.py) with lossless provenance, explicit label mappings
(src/data/mappings.py), validation, and the dataset mapping report.

Version A preprocessing (src/data/prepare.py) is NOT touched.

Run (repo root, .venv):
  python src/data/prepare_unified.py

Outputs:
  data/processed/unified_records.parquet
  artifacts/results/dataset_mapping_report.json
  artifacts/results/dataset_mapping_report.csv
  artifacts/results/dataset_license_manifest.json

Raw datasets are downloaded on demand into gitignored data/raw/.
FaithBench (CC BY-NC-SA) and RAGTruth official files are never committed.
"""

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import (  # noqa: E402
    EMPTY_META,
    EMPTY_SPANS,
    LABEL_MAPPING_VERSION,
    SCHEMA_VERSION,
    UNIFIED_COLUMNS,
    frame_fingerprint,
    json_dumps,
    sha256_text,
    validate_unified_df,
)
from src.data.mappings import (  # noqa: E402
    FAITHBENCH_PRIMARY_CLASSES,
    faithbench_label,
    faithbench_label_sensitivity,
    faithbench_severity,
    normalize_faithbench_label,
    ragtruth_label_from_spans,
)
from src.data.registry import license_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_unified")

PROCESSED_PARQUET = ROOT / "data" / "processed" / "unified_records.parquet"
REPORT_JSON = ROOT / "artifacts" / "results" / "dataset_mapping_report.json"
REPORT_CSV = ROOT / "artifacts" / "results" / "dataset_mapping_report.csv"
LICENSE_MANIFEST = ROOT / "artifacts" / "results" / "dataset_license_manifest.json"

RAGTRUTH_TASK_MAP = {"QA": "qa", "Summary": "summarization", "Data2txt": "data_to_text"}
RAGTRUTH_DOMAIN_MAP = {"CNN/DM": "cnn_dm", "Recent News": "recent_news", "MARCO": "marco", "Yelp": "yelp"}


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return s or "other"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _span_is_valid(answer: str, span: dict) -> bool:
    start, end = span.get("start"), span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end > len(answer) or start > end:
        return False
    text = span.get("text")
    if text is None:
        return True
    return answer[start:end] == text


# --------------------------------------------------------------------------
# Per-dataset canonical builders
# --------------------------------------------------------------------------

def build_halueval_canonical(df_with_split) -> pd.DataFrame:
    """Canonical rows from the Version A prepared frame (already group-split)."""
    rows = []
    for _, r in df_with_split.iterrows():
        rows.append({
            "sample_id": f"halueval:{r['sample_id']}",
            "source_dataset": "halueval",
            "source_group_id": f"halueval:q_{int(r['item_idx'])}",
            "task": "qa",
            "domain": "halueval",
            "question": r["question"],
            "context": r["context"],
            "answer": r["answer"],
            "label": int(r["label"]),
            "span_annotations": EMPTY_SPANS,
            "generator_model": "",
            "official_split": "",
            "experiment_split": r["split"],
            "quality": "",
            "native_record_id": str(int(r["item_idx"])),
            "native_metadata": EMPTY_META,
            "label_mapping_version": LABEL_MAPPING_VERSION,
        })
    return pd.DataFrame(rows)


def build_ragtruth_canonical(responses, sources) -> tuple:
    """Canonical rows + exclusions from official response.jsonl/source_info.jsonl."""
    rows = []
    exclusions = []
    n_invalid_spans = 0
    span_type_counts = {}

    for r in responses:
        source_id = str(r["source_id"])
        src = sources.get(source_id)
        native_id = str(r["id"])
        if src is None:
            exclusions.append({"native_id": native_id, "reason": "missing_source"})
            continue
        task = RAGTRUTH_TASK_MAP.get(src.get("task_type"))
        if task is None:
            exclusions.append({"native_id": native_id, "reason": f"unknown_task_type:{src.get('task_type')}"})
            continue

        answer = str(r.get("response") or "")
        if not answer.strip():
            exclusions.append({"native_id": native_id, "reason": "empty_answer"})
            continue

        si = src.get("source_info")
        if task == "qa":
            question = str(si.get("question", "")) if isinstance(si, dict) else ""
            context = str(si.get("passages", "")) if isinstance(si, dict) else ""
        elif task == "summarization":
            question, context = "", str(si or "")
        else:  # data_to_text: stable JSON serialization of the structured source
            question, context = "", json_dumps(si) if isinstance(si, dict) else str(si or "")

        labels = r.get("labels") or []
        for span in labels:
            span_type_counts[span.get("label_type", "unlabeled")] = span_type_counts.get(span.get("label_type", "unlabeled"), 0) + 1
            if not _span_is_valid(answer, span):
                n_invalid_spans += 1

        rows.append({
            "sample_id": f"ragtruth:{native_id}",
            "source_dataset": "ragtruth",
            "source_group_id": f"ragtruth:{source_id}",
            "task": task,
            "domain": RAGTRUTH_DOMAIN_MAP.get(str(src.get("source")), _slug(str(src.get("source")))),
            "question": question,
            "context": context,
            "answer": answer,
            "label": int(ragtruth_label_from_spans(labels)),
            "span_annotations": json_dumps(labels),
            "generator_model": str(r.get("model") or ""),
            "official_split": str(r.get("split") or ""),
            "experiment_split": "",
            "quality": str(r.get("quality") or ""),
            "native_record_id": native_id,
            "native_metadata": json_dumps({
                "temperature": r.get("temperature"),
                "source_id": source_id,
                "source": str(src.get("source") or ""),
                "task_type": src.get("task_type"),
            }),
            "label_mapping_version": LABEL_MAPPING_VERSION,
        })

    df = pd.DataFrame(rows)
    df.attrs["exclusions"] = exclusions
    df.attrs["n_invalid_spans"] = n_invalid_spans
    df.attrs["span_type_counts"] = span_type_counts
    return df


def build_faithbench_canonical(samples) -> tuple:
    """Canonical rows + exclusions from the official FaithBench batches."""
    rows = []
    exclusions = []
    n_invalid_spans = 0
    raw_label_counts = {}

    for batch_id, s in samples:
        metadata = s.get("metadata") or {}
        annotations = s.get("annotations") or []
        summary = str(s.get("summary") or "")
        if not summary.strip():
            exclusions.append({"native_id": f"batch_{batch_id}_sample_{s.get('sample_id')}", "reason": "empty_summary"})
            continue

        for ann in annotations:
            for lab in ann.get("label") or []:
                normalized = normalize_faithbench_label(lab)
                raw_label_counts[normalized] = raw_label_counts.get(normalized, 0) + 1
                if not _span_is_valid(summary, {
                    "start": ann.get("summary_start"),
                    "end": ann.get("summary_end"),
                    "text": ann.get("summary_span"),
                }):
                    n_invalid_spans += 1

        raw_id = metadata.get("raw_sample_id")
        group = f"faithbench:raw_{raw_id}" if raw_id is not None else f"faithbench:hash_{sha256_text(s['source'])[:16]}"

        rows.append({
            "sample_id": f"faithbench:batch_{batch_id}:sample_{s['sample_id']}",
            "source_dataset": "faithbench",
            "source_group_id": group,
            "task": "summarization",
            "domain": "faithbench",
            "question": "",
            "context": str(s.get("source") or ""),
            "answer": summary,
            "label": int(faithbench_label(annotations, aggregation="worst", hallucinated_classes=FAITHBENCH_PRIMARY_CLASSES)),
            "span_annotations": json_dumps(annotations),
            "generator_model": str(metadata.get("summarizer") or ""),
            "official_split": "",
            "experiment_split": "",
            "quality": "",
            "native_record_id": f"batch_{batch_id}_sample_{s['sample_id']}",
            "native_metadata": json_dumps(metadata),
            "label_mapping_version": LABEL_MAPPING_VERSION,
        })

    df = pd.DataFrame(rows)
    df.attrs["exclusions"] = exclusions
    df.attrs["n_invalid_spans"] = n_invalid_spans
    df.attrs["raw_label_counts"] = raw_label_counts
    return df


# --------------------------------------------------------------------------
# Raw data loading (downloads on demand)
# --------------------------------------------------------------------------

def ensure_raw_data():
    from src.data.download import download_halueval_qa
    from src.data.download_faithbench import download_faithbench
    from src.data.download_ragtruth import download_ragtruth_official

    download_halueval_qa()
    download_ragtruth_official()
    download_faithbench()


def load_raw() -> dict:
    from src.data.download_faithbench import load_faithbench_samples
    from src.data.download_ragtruth import load_ragtruth_official
    from src.data.prepare import load_and_parse_raw_data, group_split_by_item

    halueval_raw = load_and_parse_raw_data()
    halueval_split, split_report = group_split_by_item(halueval_raw)
    responses, sources = load_ragtruth_official()
    faithbench_samples = load_faithbench_samples()
    return {
        "halueval_split": halueval_split,
        "split_report": split_report,
        "ragtruth_responses": responses,
        "ragtruth_sources": sources,
        "faithbench_samples": faithbench_samples,
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def _counts(df: pd.DataFrame, col: str) -> dict:
    """Value counts with empty strings reported under the 'none' key."""
    counts = df[col].replace("", pd.NA).value_counts(dropna=False).to_dict()
    return {("none" if (k is None or (isinstance(k, float) and k != k)) else str(k)): int(v) for k, v in counts.items()}


def _dataset_stats(df: pd.DataFrame) -> dict:
    stats = {
        "n_rows": int(len(df)),
        "n_groups": int(df["source_group_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "positive_rate": round(float(df["label"].mean()), 4) if len(df) else 0.0,
        "task_counts": {str(k): int(v) for k, v in df["task"].value_counts().to_dict().items()},
        "domain_counts": {str(k): int(v) for k, v in df["domain"].value_counts().to_dict().items()},
        "official_split_counts": _counts(df, "official_split") if len(df) else {},
        "generator_model_counts": _counts(df, "generator_model") if len(df) else {},
        "quality_counts": _counts(df, "quality") if len(df) else {},
        "experiment_split_counts": _counts(df, "experiment_split") if len(df) else {},
        "empty_question": int((df["question"].fillna("").astype(str).str.strip() == "").sum()),
        "empty_context": int((df["context"].fillna("").astype(str).str.strip() == "").sum()),
    }
    return stats


def build_report(df: pd.DataFrame, attrs: dict) -> dict:
    groups = df.groupby("source_group_id")["source_dataset"].nunique()
    report = {
        "schema_version": SCHEMA_VERSION,
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint_sha256": frame_fingerprint(df.sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)),
        "groups_spanning_datasets": int((groups > 1).sum()),
        "datasets": {},
    }
    for name in ("halueval", "ragtruth", "faithbench"):
        sub = df[df["source_dataset"] == name]
        stats = _dataset_stats(sub)
        attrs_for = attrs.get(name, {})
        stats["excluded_records"] = attrs_for.get("exclusions", [])
        stats["n_excluded"] = len(stats["excluded_records"])
        stats["n_invalid_spans"] = attrs_for.get("n_invalid_spans", 0)
        if name == "ragtruth":
            stats["span_label_type_distribution"] = attrs_for.get("span_type_counts", {})
        if name == "faithbench":
            stats["raw_label_distribution"] = attrs_for.get("raw_label_counts", {})
            sens_counts = {"labels_1": {}, "labels_0": {}}
            for s in sub["span_annotations"].tolist():
                annotations = json.loads(s)
                for cfg, val in faithbench_label_sensitivity(annotations).items():
                    sens_counts[f"labels_{val}"][cfg] = sens_counts[f"labels_{val}"].get(cfg, 0) + 1
            stats["label_sensitivity"] = {
                cfg: {
                    "n_positive": int(sens_counts["labels_1"].get(cfg, 0)),
                    "n_negative": int(sens_counts["labels_0"].get(cfg, 0)),
                }
                for cfg in faithbench_label_sensitivity([]).keys()
            }
        report["datasets"][name] = stats

    report["raw_files"] = {}
    for path in sorted(ROOT.glob("data/raw/halueval/*.json")) + sorted(ROOT.glob("data/raw/ragtruth_official/*.jsonl")) + sorted(ROOT.glob("data/raw/faithbench/batch_*.json")):
        report["raw_files"][str(path.relative_to(ROOT))] = {
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return report


def build_report_csv(report: dict) -> pd.DataFrame:
    rows = []
    for name, stats in report["datasets"].items():
        for key, value in stats.items():
            if key in ("excluded_records",):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"dataset": name, "stat": key, "value": value})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ensure_raw_data()
    raw = load_raw()

    halueval_df = build_halueval_canonical(raw["halueval_split"])
    ragtruth_df = build_ragtruth_canonical(raw["ragtruth_responses"], raw["ragtruth_sources"])
    faithbench_df = build_faithbench_canonical(raw["faithbench_samples"])

    logger.info(
        f"Canonical rows: halueval={len(halueval_df)} ragtruth={len(ragtruth_df)} "
        f"faithbench={len(faithbench_df)}"
    )
    for name, df_ in (("ragtruth", ragtruth_df), ("faithbench", faithbench_df)):
        ex = df_.attrs.get("exclusions", [])
        if ex:
            logger.warning(f"{name}: {len(ex)} excluded records: {json.dumps(ex[:5])}")

    df = pd.concat([halueval_df, ragtruth_df, faithbench_df], ignore_index=True)
    df = df.sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)
    validate_unified_df(df)
    logger.info("Canonical schema validation passed.")

    os.makedirs(PROCESSED_PARQUET.parent, exist_ok=True)
    df.to_parquet(PROCESSED_PARQUET, index=False)
    logger.info(f"Saved {PROCESSED_PARQUET} ({len(df)} rows)")

    attrs = {
        "halueval": {"exclusions": [], "n_invalid_spans": 0},
        "ragtruth": {
            "exclusions": ragtruth_df.attrs.get("exclusions", []),
            "n_invalid_spans": ragtruth_df.attrs.get("n_invalid_spans", 0),
            "span_type_counts": ragtruth_df.attrs.get("span_type_counts", {}),
        },
        "faithbench": {
            "exclusions": faithbench_df.attrs.get("exclusions", []),
            "n_invalid_spans": faithbench_df.attrs.get("n_invalid_spans", 0),
            "raw_label_counts": faithbench_df.attrs.get("raw_label_counts", {}),
        },
    }
    report = build_report(df, attrs)
    os.makedirs(REPORT_JSON.parent, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_CSV.parent.mkdir(exist_ok=True)
    build_report_csv(report).to_csv(REPORT_CSV, index=False)
    LICENSE_MANIFEST.write_text(json.dumps(license_manifest(), indent=2), encoding="utf-8")
    logger.info(f"Reports written: {REPORT_JSON}, {REPORT_CSV}, {LICENSE_MANIFEST}")

    for name, stats in report["datasets"].items():
        logger.info(
            f"{name}: {stats['n_rows']} rows / {stats['n_groups']} groups, "
            f"label1={stats['label_counts'].get('1', 0)}, "
            f"positive_rate={stats['positive_rate']}, excluded={stats['n_excluded']}"
        )


if __name__ == "__main__":
    main()
