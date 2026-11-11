"""P0 author-review ingestion and agreement report.

The P0 package ships AI pre-labels for three sheets and a 60-item blind overlap.
This script is the bridge between the authors' completed review and the paper.
It never edits the manuscript. It writes a JSON summary, a Markdown report, and
a small LaTeX snippet whose numbers can be copied into the manuscript after the
three-author review.

Usage (repo root):
  python src/models/p0_ingest_author_review.py
  python src/models/p0_ingest_author_review.py --allow-incomplete
  python src/models/p0_ingest_author_review.py --docs-dir docs/rebuild/p0 \
      --out artifacts/results/rebuild/p0

Expected files under --docs-dir:
  p0_error_taxonomy.csv
  p0_explanation_audit.csv
  p0_claim_verdicts.csv
  p0_blind_overlap.csv

Blind overlap format. The item set has 60 unique item_ids. Each of the three
authors labels every item in a separate copy. Combine the three copies into one
file with 180 rows and an `annotator_id` column (author_a/author_b/author_c), or
use three columns `author1_label`, `author2_label`, `author3_label` on 60 rows.
Both layouts are accepted.

Integrity rule: the labels originated as AI pre-annotations and were reviewed by
the three authors. This script reports agreement between the pre-label and the
authors' final label, and the inter-author agreement on the blind overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "docs" / "rebuild" / "p0"
DEFAULT_OUT = ROOT / "artifacts" / "results" / "rebuild" / "p0"

SHEETS = {
    "taxonomy": "p0_error_taxonomy.csv",
    "explanation": "p0_explanation_audit.csv",
    "claims": "p0_claim_verdicts.csv",
}
BLIND_FILE = "p0_blind_overlap.csv"
BLIND_MERGED = "p0_blind_overlap_merged.csv"

VERIFIED_STATES = {
    "author_verified",
    "author_corrected",
    "verified",
    "adjudicated",
    "final",
    "complete",
    "completed",
    "done",
}

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "grounded": 0,
    "correct": 0,
    "faithful": 0,
    "not_hallucinated": 0,
    "hallucinated": 1,
    "incorrect": 1,
    "unfaithful": 1,
}


def norm_label(value):
    """Normalize a label without guessing. Unknown strings are kept as-is."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    return text


def norm_verdict(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    aliases = {
        "support": "supported",
        "supports": "supported",
        "entailed": "supported",
        "contradict": "contradicted",
        "contradicts": "contradicted",
        "not_supported": "unsupported",
        "no_support": "unsupported",
        "unknown": "unsure",
        "ambiguous": "unsure",
    }
    return aliases.get(text, text)


def norm_any(value):
    """Normalize a label or a verdict; taxonomy uses grounded/hallucinated."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    return norm_verdict(text)


def is_verified(series: pd.Series) -> bool:
    values = {str(v).strip().lower() for v in series.dropna().unique()}
    return bool(values) and values.issubset(VERIFIED_STATES)


def cohen_kappa(a, b) -> float | None:
    """Cohen's kappa for two label sequences, complete cases only."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs}, key=str)
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(pairs)
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in pairs:
        matrix[index[x], index[y]] += 1
    observed = np.trace(matrix) / n
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    expected = float(np.dot(row, col))
    if math.isclose(expected, 1.0):
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def fleiss_kappa(ratings: list[list]) -> float | None:
    """Fleiss' kappa for N items x n raters, complete cases only."""
    complete = [r for r in ratings if all(v is not None for v in r)]
    if len(complete) < 2 or len(complete[0]) < 2:
        return None
    n_items = len(complete)
    n_raters = len(complete[0])
    labels = sorted({v for row in complete for v in row}, key=str)
    index = {lab: i for i, lab in enumerate(labels)}
    counts = np.zeros((n_items, len(labels)), dtype=float)
    for i, row in enumerate(complete):
        for value in row:
            counts[i, index[value]] += 1
    p_i = (np.sum(counts * (counts - 1.0), axis=1)) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_i))
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j * p_j))
    if math.isclose(p_e, 1.0):
        return 1.0
    return float((p_bar - p_e) / (1.0 - p_e))


def agreement_summary(df: pd.DataFrame, ai_col: str, human_col: str) -> dict:
    ai = [norm_any(v) for v in df[ai_col]]
    human = [norm_any(v) for v in df[human_col]]
    pairs = [(a, h) for a, h in zip(ai, human) if a is not None and h is not None]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None, "corrected": None}
    agree = sum(1 for a, h in pairs if a == h)
    corrected = sum(1 for a, h in pairs if a != h)
    return {
        "n": len(pairs),
        "agreement": agree / len(pairs),
        "kappa": cohen_kappa([a for a, _ in pairs], [h for _, h in pairs]),
        "corrected": corrected,
        "corrected_rate": corrected / len(pairs),
        "ai_distribution": pd.Series([a for a, _ in pairs]).value_counts().to_dict(),
        "human_distribution": pd.Series([h for _, h in pairs]).value_counts().to_dict(),
    }


def read_blind(path: Path) -> tuple[dict, list[str]]:
    """Return per-sheet item->ratings and a list of layout warnings."""
    warnings: list[str] = []
    if not path.exists():
        return {}, [f"missing blind file: {path}"]
    df = pd.read_csv(path)
    if df.empty:
        return {}, [f"blind file is empty: {path}"]

    # Layout A: one row per item, three explicit author columns.
    # Accept author1_label/author2_label/author3_label and the guide's a1/a2/a3.
    author_cols = [
        c for c in df.columns
        if (c.lower().startswith("author") and c.lower().endswith("_label"))
        or c.lower() in {"a1", "a2", "a3"}
    ]
    if len(author_cols) >= 2 and "annotator_id" not in df.columns:
        author_cols = sorted(author_cols, key=lambda c: c.lower())
        ratings = {}
        for _, row in df.iterrows():
            item = str(row.get("item_id", ""))
            sheet = str(row.get("sheet", "unknown")).strip().lower()
            values = [norm_any(row[c]) for c in author_cols]
            ratings.setdefault(sheet, {})[item] = values
        return ratings, warnings

    # Layout B: one row per (item, annotator).
    required = {"item_id", "sheet", "annotator_id", "author_label"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"blind file missing columns: {sorted(missing)}")
        return {}, warnings
    ratings = {}
    for (sheet, item), group in df.groupby(["sheet", "item_id"], dropna=False):
        sheet_key = str(sheet).strip().lower()
        values = [norm_any(v) for v in group["author_label"]]
        ratings.setdefault(sheet_key, {})[str(item)] = values
    for sheet, items in ratings.items():
        sizes = {len(v) for v in items.values()}
        if sizes != {3}:
            warnings.append(f"{sheet}: rating counts per item are {sorted(sizes)}, expected 3")
    return ratings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write a report even if some rows are still needs_author_verification")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "schema": "p0-author-review-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(args.docs_dir),
        "sheets": {},
        "blind": {},
        "warnings": [],
        "complete": True,
    }

    for name, filename in SHEETS.items():
        path = args.docs_dir / filename
        if not path.exists():
            summary["warnings"].append(f"missing sheet: {path}")
            summary["complete"] = False
            continue
        df = pd.read_csv(path)
        sheet = {
            "file": filename,
            "rows": int(len(df)),
            "verified": is_verified(df["review_status"]) if "review_status" in df else False,
            "review_status": df["review_status"].value_counts(dropna=False).to_dict()
            if "review_status" in df else {},
        }
        if not sheet["verified"]:
            summary["complete"] = False
        # Taxonomy: the AI category is primary_category and the author writes a
        # category code into author_label. final_label, when used, is the binary
        # benchmark decision, so it is reported separately.
        if name == "taxonomy":
            ai_col = "ai_suggested"
            human_col = "author_label"
            sheet["label_agreement"] = agreement_summary(df, "label", "final_label") \
                if "final_label" in df and df["final_label"].notna().any() else None
        else:
            ai_col = "ai_suggested"
            human_col = "final_label" if "final_label" in df and df["final_label"].notna().any() else "author_label"
        sheet["ai_column"] = ai_col
        sheet["human_column"] = human_col
        sheet["agreement"] = agreement_summary(df, ai_col, human_col)
        sheet["blank_final_label"] = int(df["final_label"].isna().sum()) if "final_label" in df else len(df)
        sheet["blank_annotator_id"] = int(df["annotator_id"].isna().sum()) if "annotator_id" in df else None
        summary["sheets"][name] = sheet

    blind_path = args.docs_dir / BLIND_MERGED
    if not blind_path.exists():
        blind_path = args.docs_dir / BLIND_FILE
        summary["warnings"].append(
            f"merged blind file not found; reading {BLIND_FILE}. "
            "Fleiss' kappa requires p0_blind_overlap_merged.csv with three raters."
        )
    blind_ratings, blind_warnings = read_blind(blind_path)
    summary["blind_file"] = str(blind_path)
    summary["warnings"].extend(blind_warnings)
    for sheet_name, items in blind_ratings.items():
        ratings = list(items.values())
        n_items = len(ratings)
        n_raters = len(ratings[0]) if ratings else 0
        summary["blind"][sheet_name] = {
            "items": n_items,
            "raters": n_raters,
            "fleiss_kappa": fleiss_kappa(ratings),
            "mean_pairwise_agreement": None,
        }
        # Mean pairwise agreement over raters, complete items only.
        complete = [r for r in ratings if all(v is not None for v in r)]
        if complete and n_raters >= 2:
            pair_scores = []
            for i in range(n_raters):
                for j in range(i + 1, n_raters):
                    pair_scores.append(np.mean([r[i] == r[j] for r in complete]))
            summary["blind"][sheet_name]["mean_pairwise_agreement"] = float(np.mean(pair_scores))

    if summary["blind"]:
        kappas = [v["fleiss_kappa"] for v in summary["blind"].values() if v["fleiss_kappa"] is not None]
        summary["blind_overall"] = {
            "sheets": len(summary["blind"]),
            "mean_fleiss_kappa": float(np.mean(kappas)) if kappas else None,
        }

    three_raters = bool(summary["blind"]) and all(
        value.get("raters", 0) >= 3 and value.get("fleiss_kappa") is not None
        for value in summary["blind"].values()
    )
    if not three_raters:
        summary["warnings"].append(
            "three-author blind overlap is incomplete: every sheet needs three raters "
            "and a non-null Fleiss kappa before the review can be marked complete"
        )
    summary["complete"] = bool(
        summary["complete"]
        and all(v.get("verified") for v in summary["sheets"].values())
        and three_raters
    )

    out_json = args.out / "author_review_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Markdown report.
    lines = [
        "# P0 author-review summary",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Complete: **{summary['complete']}**",
        "",
        "The labels originated as AI pre-annotations and were reviewed by the three",
        "authors. Agreement below compares the pre-label with the authors' final",
        "label; the blind overlap is the inter-author agreement.",
        "",
        "## Sheets",
        "",
        "| Sheet | Rows | Verified | Blank final | AI-author agreement | Cohen kappa | Corrected |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, sheet in summary["sheets"].items():
        agr = sheet.get("agreement", {})
        lines.append(
            f"| {name} | {sheet['rows']} | {sheet['verified']} | {sheet.get('blank_final_label')} | "
            f"{fmt(agr.get('agreement'))} | {fmt(agr.get('kappa'))} | {fmt(agr.get('corrected_rate'))} |"
        )
    lines += ["", "## Blind overlap (three authors)", ""]
    if summary["blind"]:
        lines += [
            "| Sheet | Items | Raters | Fleiss kappa | Mean pairwise agreement |",
            "|---|---:|---:|---:|---:|",
        ]
        for sheet_name, value in summary["blind"].items():
            lines.append(
                f"| {sheet_name} | {value['items']} | {value['raters']} | "
                f"{fmt(value['fleiss_kappa'])} | {fmt(value['mean_pairwise_agreement'])} |"
            )
        overall = summary.get("blind_overall", {}).get("mean_fleiss_kappa")
        lines += ["", f"Mean Fleiss kappa across sheets: **{fmt(overall)}**."]
    else:
        lines.append("No blind overlap file was readable.")
    if summary["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in summary["warnings"]]
    lines += [
        "",
        "## Paper wording",
        "",
        "Use: \"three-author blinded review with AI pre-annotation\".",
        "Do not use: \"independent human annotation\" or \"two human annotators\".",
        "",
        "## Next step",
        "",
        "Copy the agreement and kappa values into the manuscript's audit paragraph.",
        "Do not copy the AI pre-label distribution as a human result; copy the authors' final distribution.",
    ]
    (args.out / "author_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX snippet with conservative placeholders.
    snippet = [
        "% Generated by src/models/p0_ingest_author_review.py",
        "% Fill only after every sheet has review_status=author_verified.",
    ]
    for name, sheet in summary["sheets"].items():
        agr = sheet.get("agreement", {})
        snippet.append(
            f"% {name}: n={agr.get('n')} agreement={fmt(agr.get('agreement'))} "
            f"kappa={fmt(agr.get('kappa'))} corrected={fmt(agr.get('corrected_rate'))}"
        )
    for sheet_name, value in summary["blind"].items():
        snippet.append(
            f"% blind/{sheet_name}: fleiss={fmt(value['fleiss_kappa'])} "
            f"pairwise={fmt(value['mean_pairwise_agreement'])}"
        )
    (args.out / "author_review_snippet.tex").write_text("\n".join(snippet) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {args.out / 'author_review_report.md'}")
    print(f"wrote {args.out / 'author_review_snippet.tex'}")
    print(f"complete={summary['complete']}")
    if not summary["complete"] and not args.allow_incomplete:
        print("review is incomplete; rerun with --allow-incomplete to keep a draft report", file=sys.stderr)
        return 1
    return 0


def fmt(value):
    if value is None:
        return "---"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
