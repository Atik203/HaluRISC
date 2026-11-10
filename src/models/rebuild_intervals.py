"""WP3 - per-seed and two-level (hierarchical) bootstrap intervals.

Every headline delta is recomputed with 2,000 draws of a two-level bootstrap:
groups are resampled with replacement within a seed, then seeds are resampled
with replacement.  Group ids are HaluEval ``item_idx`` (the two prompt variants
of one item stay together) and external ``source_group_id``.

Comparisons
-----------
* HaluEval test: m3-m0 (F1, AUROC), m2-m1 (F1, AUROC), m3-m1 (F1, AUROC)
* RAGTruth QA test: m3-m0 (F1, AUROC)
* RAGTruth all-test: m3-m0 (F1, AUROC)

Both the rebuilt fixed-hp models and the frozen tuned-hp B6 models are scored
on the same rows and groups.  The paired contrast (fixed minus tuned) uses the
same resamples, so the difference has its own 95% CI.

Outputs (artifacts/results/rebuild/intervals/):
  intervals.csv, intervals_contrast.csv, intervals_report.json, REPORT.md

Run (repo root, .venv):
  python src/models/rebuild_intervals.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.models.config import SEEDS
from src.models.rebuild_common import (
    B6_PRED,
    FIXED_HP_PRED,
    INTERVALS_DIR,
    N_BOOT_INTERVALS,
    THRESHOLD,
    draw_group_row_indices,
    f1_and_auroc,
    group_frame,
    load_predictions,
    stable_seed,
    summarize_bootstrap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_intervals")

SOURCES = {
    "fixed_hp": FIXED_HP_PRED,
    "frozen_b6_tuned": B6_PRED,
}
DATASET_VARIANTS = {
    "halueval_test": ("m0", "m1", "m2", "m3"),
    "ragtruth_qa_test": ("m0", "m3"),
    "ragtruth_all_test": ("m0", "m3"),
}
DATASET_COMPARISONS = {
    "halueval_test": (("m3", "m0"), ("m2", "m1"), ("m3", "m1")),
    "ragtruth_qa_test": (("m3", "m0"),),
    "ragtruth_all_test": (("m3", "m0"),),
}
DATASET_ORDER = ("halueval_test", "ragtruth_qa_test", "ragtruth_all_test")
METRICS = ("f1", "auroc")
CI_LEVEL = 0.95

# The six results that carry the rebuild narrative (comparison, dataset, metric).
HEADLINE = (
    ("m3", "m0", "halueval_test", "f1"),
    ("m3", "m0", "halueval_test", "auroc"),
    ("m2", "m1", "halueval_test", "f1"),
    ("m3", "m1", "halueval_test", "f1"),
    ("m3", "m0", "ragtruth_qa_test", "auroc"),
    ("m3", "m0", "ragtruth_all_test", "f1"),
)

INTERVAL_COLUMNS = [
    "source", "dataset", "comparison", "metric", "seed", "n_rows", "n_groups",
    "n_seeds", "point_delta", "ci_low", "ci_high", "ci_level", "n_boot",
]
CONTRAST_COLUMNS = [
    "dataset", "comparison", "metric", "seed", "n_rows", "n_groups",
    "delta_fixed_hp", "delta_frozen_b6_tuned", "difference_fixed_minus_tuned",
    "ci_low", "ci_high", "ci_level", "n_boot",
]


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def build_matrix(dataset: str, sources: dict) -> dict:
    """Wide dataset matrix with one score column per source/seed/variant.

    Rows are sorted by group id, so groups are contiguous for the cluster
    bootstrap.  Labels are checked against every prediction file.
    """
    merged = group_frame(dataset)
    variants = DATASET_VARIANTS[dataset]

    for source, path in sources.items():
        preds = load_predictions(path)
        subset = preds[preds["dataset"] == dataset]
        if subset.empty:
            raise ValueError(f"{path} has no rows for {dataset}")
        for seed in SEEDS:
            for variant in variants:
                block = subset[
                    (subset["seed"] == seed) & (subset["variant"] == variant)
                ][["sample_id", "label", "score"]]
                if block.empty:
                    raise ValueError(f"missing {source} seed {seed} {variant} {dataset}")
                block = block.rename(
                    columns={
                        "label": f"label__{source}__{seed}__{variant}",
                        "score": f"score__{source}__{seed}__{variant}",
                    }
                )
                merged = merged.merge(block, on="sample_id", how="left", validate="one_to_one")
                score_col = f"score__{source}__{seed}__{variant}"
                label_col = f"label__{source}__{seed}__{variant}"
                if merged[score_col].isna().any():
                    raise ValueError(f"{dataset}: missing scores for {source} {seed} {variant}")
                if not np.array_equal(
                    merged["label"].to_numpy(dtype=int),
                    merged[label_col].to_numpy(dtype=int),
                ):
                    raise ValueError(f"{dataset}: label mismatch for {source} {seed} {variant}")

    merged = merged.sort_values(["group_id", "sample_id"]).reset_index(drop=True)
    group_sizes = merged.groupby("group_id", sort=False).size().to_numpy(dtype=int)
    codes = pd.factorize(merged["group_id"], sort=False)[0]
    expected = np.repeat(np.arange(len(group_sizes)), group_sizes)
    if not np.array_equal(codes, expected):
        raise ValueError(f"{dataset}: groups are not contiguous after sorting")

    scores = {}
    for source in sources:
        scores[source] = {}
        for seed in SEEDS:
            scores[source][seed] = {
                variant: merged[f"score__{source}__{seed}__{variant}"].to_numpy(dtype=float)
                for variant in variants
            }

    return {
        "y": merged["label"].to_numpy(dtype=int),
        "group_sizes": group_sizes,
        "n_groups": int(len(group_sizes)),
        "n_rows": int(len(merged)),
        "scores": scores,
    }


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_dataset(dataset: str, matrix: dict) -> dict:
    """Run the within-seed and two-level bootstraps for one dataset.

    Returns point estimates and bootstrap distributions keyed by
    ``(source, comparison, metric, seed_or_pooled)`` plus the paired
    fixed-minus-tuned contrast distributions.
    """
    y = matrix["y"]
    group_sizes = matrix["group_sizes"]
    scores = matrix["scores"]
    variants = DATASET_VARIANTS[dataset]
    comparisons = DATASET_COMPARISONS[dataset]

    point = {}
    for source in SOURCES:
        for seed in SEEDS:
            for variant in variants:
                point[(source, seed, variant)] = f1_and_auroc(y, scores[source][seed][variant])

    point_seed = {}
    for source in SOURCES:
        for a, b in comparisons:
            for metric_index, metric in enumerate(METRICS):
                for seed in SEEDS:
                    point_seed[(source, a, b, metric, seed)] = float(
                        point[(source, seed, a)][metric_index]
                        - point[(source, seed, b)][metric_index]
                    )

    point_pooled = {}
    for source in SOURCES:
        for a, b in comparisons:
            for metric_index, metric in enumerate(METRICS):
                point_pooled[(source, a, b, metric)] = float(
                    np.mean([
                        point_seed[(source, a, b, metric, seed)] for seed in SEEDS
                    ])
                )

    within_values: dict[tuple, list[float]] = {}
    pooled_values: dict[tuple, list[float]] = {}
    contrast_within: dict[tuple, list[float]] = {}
    contrast_pooled: dict[tuple, list[float]] = {}

    for metric_index, metric in enumerate(METRICS):
        for source in SOURCES:
            for a, b in comparisons:
                for seed in SEEDS:
                    within_values[(source, a, b, metric, seed)] = []
                pooled_values[(source, a, b, metric)] = []
        for a, b in comparisons:
            for seed in SEEDS:
                contrast_within[(a, b, metric, seed)] = []
            contrast_pooled[(a, b, metric)] = []

    # ---- Within-seed cluster bootstrap ----
    for seed in SEEDS:
        rng = np.random.default_rng(
            stable_seed("intervals_within", dataset, seed)
        )
        for _ in range(N_BOOT_INTERVALS):
            idx = draw_group_row_indices(group_sizes, rng)
            y_boot = y[idx]
            cache = {}
            for source in SOURCES:
                for variant in variants:
                    cache[(source, variant)] = f1_and_auroc(
                        y_boot, scores[source][seed][variant][idx]
                    )
            for source in SOURCES:
                for a, b in comparisons:
                    for metric_index, metric in enumerate(METRICS):
                        within_values[(source, a, b, metric, seed)].append(
                            cache[(source, a)][metric_index] - cache[(source, b)][metric_index]
                        )
            for a, b in comparisons:
                for metric_index, metric in enumerate(METRICS):
                    contrast_within[(a, b, METRICS[metric_index], seed)].append(
                        (cache[("fixed_hp", a)][metric_index] - cache[("fixed_hp", b)][metric_index])
                        - (
                            cache[("frozen_b6_tuned", a)][metric_index]
                            - cache[("frozen_b6_tuned", b)][metric_index]
                        )
                    )

    # ---- Two-level bootstrap: groups within seed, then seeds ----
    pooled_rng = np.random.default_rng(stable_seed("intervals_pooled", dataset))
    for _ in range(N_BOOT_INTERVALS):
        chosen = pooled_rng.integers(0, len(SEEDS), len(SEEDS))
        per_seed_deltas = {
            (source, a, b, metric): []
            for source in SOURCES
            for a, b in comparisons
            for metric in METRICS
        }
        for seed_position in chosen:
            seed = SEEDS[int(seed_position)]
            idx = draw_group_row_indices(group_sizes, pooled_rng)
            y_boot = y[idx]
            cache = {}
            for source in SOURCES:
                for variant in variants:
                    cache[(source, variant)] = f1_and_auroc(
                        y_boot, scores[source][seed][variant][idx]
                    )
            for source in SOURCES:
                for a, b in comparisons:
                    for metric_index, metric in enumerate(METRICS):
                        per_seed_deltas[(source, a, b, metric)].append(
                            cache[(source, a)][metric_index] - cache[(source, b)][metric_index]
                        )
        for source in SOURCES:
            for a, b in comparisons:
                for metric in METRICS:
                    pooled_values[(source, a, b, metric)].append(
                        float(np.mean(per_seed_deltas[(source, a, b, metric)]))
                    )
        for a, b in comparisons:
            for metric in METRICS:
                contrast_pooled[(a, b, metric)].append(
                    float(np.mean(per_seed_deltas[("fixed_hp", a, b, metric)]))
                    - float(np.mean(per_seed_deltas[("frozen_b6_tuned", a, b, metric)]))
                )
        if len(pooled_values[("fixed_hp", comparisons[0][0], comparisons[0][1], METRICS[0])]) % 250 == 0:
            logger.info(
                "%s: %d/%d two-level draws",
                dataset,
                len(pooled_values[("fixed_hp", comparisons[0][0], comparisons[0][1], METRICS[0])]),
                N_BOOT_INTERVALS,
            )

    def summarize(values: list[float]) -> dict:
        return summarize_bootstrap(values)

    within = {
        key: {"point": point_seed[key], **summarize(values)}
        for key, values in within_values.items()
    }
    pooled = {
        key: {"point": point_pooled[key], **summarize(values)}
        for key, values in pooled_values.items()
    }
    within_contrast = {
        key: {
            "point": point_seed[("fixed_hp",) + key] - point_seed[("frozen_b6_tuned",) + key],
            **summarize(values),
        }
        for key, values in contrast_within.items()
    }
    pooled_contrast = {
        key: {
            "point": point_pooled[("fixed_hp",) + key] - point_pooled[("frozen_b6_tuned",) + key],
            **summarize(values),
        }
        for key, values in contrast_pooled.items()
    }

    return {
        "within": within,
        "pooled": pooled,
        "contrast_within": within_contrast,
        "contrast_pooled": pooled_contrast,
        "n_rows": matrix["n_rows"],
        "n_groups": matrix["n_groups"],
    }


# --------------------------------------------------------------------------
# Tables and report
# --------------------------------------------------------------------------
def build_tables(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows = []
    contrast_rows = []
    for dataset in DATASET_ORDER:
        block = results[dataset]
        for source in SOURCES:
            for a, b in DATASET_COMPARISONS[dataset]:
                for metric in METRICS:
                    for seed in SEEDS:
                        stats = block["within"][(source, a, b, metric, seed)]
                        interval_rows.append(
                            {
                                "source": source,
                                "dataset": dataset,
                                "comparison": f"{a}-{b}",
                                "metric": metric,
                                "seed": str(seed),
                                "n_rows": block["n_rows"],
                                "n_groups": block["n_groups"],
                                "n_seeds": 1,
                                "point_delta": stats["point"],
                                "ci_low": stats["q025"],
                                "ci_high": stats["q975"],
                                "ci_level": CI_LEVEL,
                                "n_boot": N_BOOT_INTERVALS,
                            }
                        )
                    stats = block["pooled"][(source, a, b, metric)]
                    interval_rows.append(
                        {
                            "source": source,
                            "dataset": dataset,
                            "comparison": f"{a}-{b}",
                            "metric": metric,
                            "seed": "pooled",
                            "n_rows": block["n_rows"],
                            "n_groups": block["n_groups"],
                            "n_seeds": len(SEEDS),
                            "point_delta": stats["point"],
                            "ci_low": stats["q025"],
                            "ci_high": stats["q975"],
                            "ci_level": CI_LEVEL,
                            "n_boot": N_BOOT_INTERVALS,
                        }
                    )

        for a, b in DATASET_COMPARISONS[dataset]:
            for metric in METRICS:
                for seed in SEEDS:
                    stats = block["contrast_within"][(a, b, metric, seed)]
                    contrast_rows.append(
                        {
                            "dataset": dataset,
                            "comparison": f"{a}-{b}",
                            "metric": metric,
                            "seed": str(seed),
                            "n_rows": block["n_rows"],
                            "n_groups": block["n_groups"],
                            "delta_fixed_hp": block["within"][("fixed_hp", a, b, metric, seed)]["point"],
                            "delta_frozen_b6_tuned": block["within"][
                                ("frozen_b6_tuned", a, b, metric, seed)
                            ]["point"],
                            "difference_fixed_minus_tuned": stats["point"],
                            "ci_low": stats["q025"],
                            "ci_high": stats["q975"],
                            "ci_level": CI_LEVEL,
                            "n_boot": N_BOOT_INTERVALS,
                        }
                    )
                stats = block["contrast_pooled"][(a, b, metric)]
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{a}-{b}",
                        "metric": metric,
                        "seed": "pooled",
                        "n_rows": block["n_rows"],
                        "n_groups": block["n_groups"],
                        "delta_fixed_hp": block["pooled"][("fixed_hp", a, b, metric)]["point"],
                        "delta_frozen_b6_tuned": block["pooled"][
                            ("frozen_b6_tuned", a, b, metric)
                        ]["point"],
                        "difference_fixed_minus_tuned": stats["point"],
                        "ci_low": stats["q025"],
                        "ci_high": stats["q975"],
                        "ci_level": CI_LEVEL,
                        "n_boot": N_BOOT_INTERVALS,
                    }
                )
    return (
        pd.DataFrame(interval_rows, columns=INTERVAL_COLUMNS),
        pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_report(results: dict) -> dict:
    headline = []
    for a, b, dataset, metric in HEADLINE:
        block = results[dataset]
        fixed = block["pooled"][("fixed_hp", a, b, metric)]
        tuned = block["pooled"][("frozen_b6_tuned", a, b, metric)]
        contrast = block["contrast_pooled"][(a, b, metric)]
        headline.append(
            {
                "comparison": f"{a}-{b}",
                "dataset": dataset,
                "metric": metric,
                "fixed_hp": {
                    "per_seed": {
                        str(seed): block["within"][("fixed_hp", a, b, metric, seed)]["point"]
                        for seed in SEEDS
                    },
                    "point": fixed["point"],
                    "ci": [fixed["q025"], fixed["q975"]],
                    "sd_across_seeds": float(np.std(
                        [block["within"][("fixed_hp", a, b, metric, seed)]["point"] for seed in SEEDS],
                        ddof=1,
                    )),
                },
                "frozen_b6_tuned": {
                    "per_seed": {
                        str(seed): block["within"][("frozen_b6_tuned", a, b, metric, seed)]["point"]
                        for seed in SEEDS
                    },
                    "point": tuned["point"],
                    "ci": [tuned["q025"], tuned["q975"]],
                },
                "contrast_fixed_minus_tuned": {
                    "point": contrast["point"],
                    "ci": [contrast["q025"], contrast["q975"]],
                },
            }
        )

    return {
        "schema": "rebuild-wp3-intervals-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "script": "src/models/rebuild_intervals.py",
        "protocol": {
            "n_boot": N_BOOT_INTERVALS,
            "bootstrap": (
                "two-level: resample groups with replacement within each seed, "
                "then resample the three seeds with replacement"
            ),
            "within_seed_bootstrap": "group bootstrap, 2,000 draws per seed",
            "pooled_bootstrap": "two-level, 2,000 draws",
            "seed_resampling": "seeds 42, 123, 456 drawn with replacement (3 draws per replicate)",
            "groups": {
                "halueval_test": "item_idx (1,500 groups of 2 prompt variants)",
                "ragtruth_qa_test": "source_group_id (150 groups of 6 rows)",
                "ragtruth_all_test": "source_group_id (450 groups of 6 rows)",
            },
            "threshold": THRESHOLD,
            "ci": "percentile 95% interval",
            "sources": {
                "fixed_hp": "artifacts/results/rebuild/fixed_hp/fixed_hp_predictions.parquet",
                "frozen_b6_tuned": "artifacts/results/b6/b6_predictions.parquet",
            },
            "headline_selection": [f"{a}-{b} {metric} on {dataset}" for a, b, dataset, metric in HEADLINE],
        },
        "dataset_sizes": {
            dataset: {
                "n_rows": results[dataset]["n_rows"],
                "n_groups": results[dataset]["n_groups"],
            }
            for dataset in DATASET_ORDER
        },
        "headline": headline,
        "caveats": [
            "The frozen tuned-hp rows are read from b6_predictions.parquet; no tuned model is refit.",
            "HaluEval groups are item_idx pairs, so the two prompt variants of one item always move together.",
            "The QA and all-test datasets have only 150 and 450 groups, so their intervals are wide by construction.",
            "Bootstrap CIs describe sampling and seed variability, not the choice of feature set or calibrator.",
            "F1 uses a fixed 0.5 threshold; AUROC is threshold-free. Predicted-positive rates differ across variants, so F1 alone can reward flag-rate changes.",
            "This script does not recompute ECE; WP2 owns calibration metrics.",
        ],
    }


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown(table: pd.DataFrame, contrast: pd.DataFrame, results: dict) -> str:
    lines = [
        "# WP3 - Per-Seed and Hierarchical Intervals",
        "",
        "Generated by `src/models/rebuild_intervals.py`. Every number comes from",
        "`intervals.csv`, `intervals_contrast.csv`, and the frozen prediction files.",
        "",
        "## Protocol",
        "",
        f"- Two-level bootstrap, {N_BOOT_INTERVALS:,} draws: resample groups within",
        "  each seed, then resample seeds 42/123/456 with replacement.",
        "- HaluEval groups: `item_idx` (1,500 groups of two prompt variants).",
        "- RAGTruth QA: `source_group_id` (150 groups of six rows).",
        "- RAGTruth all-test: `source_group_id` (450 groups of six rows).",
        f"- Metrics: F1 at threshold {THRESHOLD}, AUROC. Percentile 95% CIs.",
        "- `fixed_hp` is the rebuilt WP1 ablation; `frozen_b6_tuned` is the",
        "  frozen tuned-hp B6 prediction table, read without refitting.",
        "",
        "## Commands",
        "",
        "```powershell",
        "& .venv\\Scripts\\python.exe src\\models\\rebuild_intervals.py",
        "& .venv\\Scripts\\python.exe -m pytest tests\\test_rebuild_calibration.py -v",
        "```",
        "",
        "## Headline deltas",
        "",
        "| dataset | comparison | metric | fixed 42 | fixed 123 | fixed 456 | fixed pooled | fixed 95% CI | tuned pooled | tuned 95% CI | fixed - tuned [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for a, b, dataset, metric in HEADLINE:
        fixed = results[dataset]["pooled"][("fixed_hp", a, b, metric)]
        tuned = results[dataset]["pooled"][("frozen_b6_tuned", a, b, metric)]
        diff = results[dataset]["contrast_pooled"][(a, b, metric)]
        per_seed = [
            results[dataset]["within"][("fixed_hp", a, b, metric, seed)]["point"]
            for seed in SEEDS
        ]
        lines.append(
            f"| {dataset} | {a}-{b} | {metric} | {fmt(per_seed[0])} | {fmt(per_seed[1])} | "
            f"{fmt(per_seed[2])} | {fmt(fixed['point'])} | "
            f"[{fmt(fixed['q025'])}, {fmt(fixed['q975'])}] | {fmt(tuned['point'])} | "
            f"[{fmt(tuned['q025'])}, {fmt(tuned['q975'])}] | "
            f"{fmt(diff['point'])} [{fmt(diff['q025'])}, {fmt(diff['q975'])}] |"
        )
    lines += [
        "",
        "The contrast is paired: fixed and tuned scores use the same resampled",
        "groups and the same seed draws. A CI that contains zero means the rebuild",
        "did not change the delta more than sampling noise.",
        "",
        "## Full table",
        "",
        "`intervals.csv` contains every source/dataset/comparison/metric/seed row,",
        "including the per-seed cluster-bootstrap CIs. `intervals_contrast.csv`",
        "holds the paired fixed-minus-tuned differences.",
        "",
        "## Reading the intervals",
        "",
        "- The six HaluEval comparison cells are saturated: F1 deltas are around",
        "  a thousandth of a point and AUROC deltas are in the third decimal. Their CIs are",
        "  narrow because the groups are pairs, but the effects are tiny.",
        "- RAGTruth QA AUROC is near chance. The m3-m0 delta CI is the honest",
        "  statement of whether the feature stack transfers, and it spans zero in",
        "  the rebuild.",
        "- RAGTruth QA F1 at 0.5 is not diagnostic: both variants flag almost every",
        "  row, so the positive F1 delta mostly tracks the flag rate rather than",
        "  ranking skill. The QA headline is AUROC for that reason.",
        "- The all-test F1 delta mixes summarization, data-to-text, and QA, so it",
        "  is a seen-task aggregate, not evidence of unseen-task transfer.",
        "",
        "## Caveats",
        "",
        "- CIs reflect sampling and seed variability only; they do not cover",
        "  feature-set or calibrator selection uncertainty.",
        "- F1 uses a fixed 0.5 threshold. Flag rates differ across variants, so F1",
        "  can move for reasons unrelated to ranking quality; AUROC is the",
        "  threshold-free companion.",
        "- The QA and all-test group counts are small (150 and 450), which is why",
        "  their intervals are wide even at 2,000 draws.",
        "- No test rows were used to tune or select a model or threshold.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    INTERVALS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset in DATASET_ORDER:
        logger.info("loading and checking %s", dataset)
        matrix = build_matrix(dataset, SOURCES)
        logger.info(
            "%s: %d rows, %d groups; bootstrapping %d draws",
            dataset, matrix["n_rows"], matrix["n_groups"], N_BOOT_INTERVALS,
        )
        results[dataset] = bootstrap_dataset(dataset, matrix)

    table, contrast = build_tables(results)
    table.to_csv(INTERVALS_DIR / "intervals.csv", index=False)
    contrast.to_csv(INTERVALS_DIR / "intervals_contrast.csv", index=False)

    report = build_report(results)
    (INTERVALS_DIR / "intervals_report.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )
    (INTERVALS_DIR / "REPORT.md").write_text(
        build_markdown(table, contrast, results), encoding="utf-8"
    )

    print("\nHeadline deltas (fixed_hp vs frozen tuned B6):")
    for a, b, dataset, metric in HEADLINE:
        fixed = results[dataset]["pooled"][("fixed_hp", a, b, metric)]
        tuned = results[dataset]["pooled"][("frozen_b6_tuned", a, b, metric)]
        diff = results[dataset]["contrast_pooled"][(a, b, metric)]
        print(
            f"{dataset:20s} {a}-{b} {metric:5s} | fixed {fixed['point']:+.4f} "
            f"[{fixed['q025']:+.4f}, {fixed['q975']:+.4f}] | "
            f"tuned {tuned['point']:+.4f} [{tuned['q025']:+.4f}, {tuned['q975']:+.4f}] | "
            f"diff {diff['point']:+.4f} [{diff['q025']:+.4f}, {diff['q975']:+.4f}]"
        )
    print(f"\nSaved WP3 outputs to {INTERVALS_DIR}")


if __name__ == "__main__":
    main()
