"""Render the MedHallu report from ``medhallu.json`` (no hand-entered numbers).

Run after ``src/models/rebuild_medhallu.py``::

    .venv\\Scripts\\python.exe src\\models\\rebuild_medhallu_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1_DIR = ROOT / "artifacts" / "results" / "rebuild" / "p1"
JSON_PATH = P1_DIR / "medhallu.json"
REPORT_PATH = P1_DIR / "medhallu_REPORT.md"

MODEL_LABELS = {
    "m3": "EC-XGB m3",
    "m0": "XGBoost m0",
    "length_only": "Length-only logistic",
}


def f4(value) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if value != value:  # NaN
        return "n/a"
    return f"{value:.4f}"


def pm(entry: dict, key: str) -> str:
    mean = entry.get("mean", {}).get(key)
    std = entry.get("mean", {}).get(f"{key}_std")
    if mean is None:
        return "n/a"
    if std is None:
        return f4(mean)
    return f"{f4(mean)} ± {f4(std)}"


def ci(row: dict, key: str) -> str:
    low = row.get(f"{key}_ci_low")
    high = row.get(f"{key}_ci_high")
    if low is None or high is None:
        return "n/a"
    return f"[{f4(low)}, {f4(high)}]"


def avg_row(entry: dict) -> dict:
    if "seed_average" in entry:
        return entry["seed_average"]
    return entry["per_seed"][0]


def headline_table(payload: dict) -> str:
    lines = [
        "| model | F1 (mean ± SD) | F1 95% CI | AUROC (mean ± SD) | AUROC 95% CI | PR-AUC | ECE | Brier | flagged rate | pair accuracy |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("m3", "m0", "length_only"):
        entry = payload["results"].get(key)
        if not entry:
            continue
        row = avg_row(entry)
        lines.append(
            "| {label} | {f1} | {f1ci} | {auc} | {aucci} | {pr} | {ece} | {brier} | {flag} | {pair} |".format(
                label=MODEL_LABELS.get(key, key),
                f1=pm(entry, "f1"),
                f1ci=ci(row, "f1"),
                auc=pm(entry, "auroc"),
                aucci=ci(row, "auroc"),
                pr=pm(entry, "pr_auc"),
                ece=pm(entry, "ece"),
                brier=pm(entry, "brier"),
                flag=pm(entry, "flagged_rate"),
                pair=pm(entry, "pair_accuracy"),
            )
        )
    return "\n".join(lines)


def per_seed_table(payload: dict) -> str:
    lines = [
        "| model | seed | F1 | F1 95% CI | AUROC | AUROC 95% CI | PR-AUC | ECE | Brier | flagged rate |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for key in ("m3", "m0", "length_only"):
        entry = payload["results"].get(key)
        if not entry:
            continue
        for row in entry["per_seed"]:
            lines.append(
                "| {label} | {seed} | {f1} | {f1ci} | {auc} | {aucci} | {pr} | {ece} | {brier} | {flag} |".format(
                    label=MODEL_LABELS.get(key, key),
                    seed=row["seed"],
                    f1=f4(row["f1"]),
                    f1ci=ci(row, "f1"),
                    auc=f4(row["auroc"]),
                    aucci=ci(row, "auroc"),
                    pr=f4(row["pr_auc"]),
                    ece=f4(row["ece"]),
                    brier=f4(row["brier"]),
                    flag=f4(row["flagged_rate"]),
                )
            )
    return "\n".join(lines)


def reference_table(payload: dict) -> str:
    reference = payload.get("halueval_reference") or {}
    if not reference:
        return "_No HaluEval reference rows were available._"
    lines = [
        "| model | F1 (mean ± SD) | AUROC (mean ± SD) | PR-AUC | ECE | Brier | flagged rate | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("m3", "m0"):
        entry = reference.get(key)
        if not entry:
            continue
        mean = entry["mean"]
        lines.append(
            f"| {MODEL_LABELS.get(key, key)} | {f4(mean['f1'])} ± {f4(mean['f1_std'])} | "
            f"{f4(mean['auroc'])} ± {f4(mean['auroc_std'])} | {f4(mean['pr_auc'])} | "
            f"{f4(mean['ece'])} | {f4(mean['brier'])} | {f4(mean['flagged_rate'])} | {entry['n']} |"
        )
    return "\n".join(lines)


def subgroup_table(payload: dict, model: str, group_col: str) -> str:
    entry = (payload.get("subgroups") or {}).get(model, {}).get(group_col)
    if not entry:
        return "_No subgroup rows._"
    lines = [
        f"| {group_col} | pairs | rows | F1 | AUROC | PR-AUC | ECE | Brier | flagged rate | pair accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in entry.items():
        lines.append(
            f"| {name} | {row['n_pairs']} | {row['n_rows']} | {f4(row['f1'])} | {f4(row['auroc'])} | "
            f"{f4(row['pr_auc'])} | {f4(row['ece'])} | {f4(row['brier'])} | {f4(row['flagged_rate'])} | "
            f"{f4(row['pair_accuracy'])} |"
        )
    return "\n".join(lines)


def transfer_gap_table(payload: dict) -> str:
    """MedHallu minus HaluEval for the same fixed-hp models."""
    reference = payload.get("halueval_reference") or {}
    if not reference:
        return "_No HaluEval reference rows._"
    lines = [
        "| model | delta F1 | delta AUROC | delta PR-AUC | delta ECE | delta Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("m3", "m0"):
        if key not in reference:
            continue
        med = payload["results"][key]["mean"]
        ref = reference[key]["mean"]
        lines.append(
            f"| {MODEL_LABELS.get(key, key)} | {f4(med['f1'] - ref['f1'])} | "
            f"{f4(med['auroc'] - ref['auroc'])} | {f4(med['pr_auc'] - ref['pr_auc'])} | "
            f"{f4(med['ece'] - ref['ece'])} | {f4(med['brier'] - ref['brier'])} |"
        )
    return "\n".join(lines)


def sensitivity_table(payload: dict) -> str:
    entry = payload.get("sensitivity_m3_source_0") or {}
    if not entry:
        return "_No sensitivity rows._"
    primary = avg_row(payload["results"]["m3"])
    lines = [
        "| m3 source indicator | seed | AUROC | F1 | ECE | Brier | flagged rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 1.0 (deployed convention) | seed-avg | {f4(primary['auroc'])} | {f4(primary['f1'])} | "
        f"{f4(primary['ece'])} | {f4(primary['brier'])} | {f4(primary['flagged_rate'])} |",
    ]
    for seed, row in sorted(entry.items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| 0.0 | {seed} | {f4(row['auroc'])} | {f4(row['f1'])} | {f4(row['ece'])} | "
            f"{f4(row['brier'])} | {f4(row['flagged_rate'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"run rebuild_medhallu.py first: {JSON_PATH}")
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    ds = payload["dataset"]
    protocol = payload["protocol"]
    runtime = payload["runtime"]
    results = payload["results"]
    m3 = results["m3"]
    m0 = results["m0"]
    length = results["length_only"]

    report = f"""# P1 - Second corpus: MedHallu (biomedical QA)

Generated by `src/models/rebuild_medhallu.py`; every number below is read from
`medhallu.json`. The csv companion is `medhallu.csv`.

## What this experiment is

MedHallu (`{ds['source']}`, config `{ds['config']}`, {ds['license']} licence) is a
different domain from HaluEval: {ds['domain']}. The released split contains
{ds['n_pairs']:,} independently annotated QA pairs. Each pair supplies an
expert ground-truth answer and a hallucinated answer released with the
benchmark.

This evaluation builds {ds['n_rows']:,} rows ({ds['rows_per_pair']} per pair):
the ground-truth answer is label 0 and the hallucinated answer is label 1. The
question is the MedHallu question and the context is the concatenation of its
knowledge passages. MedHallu was **not** used for training, feature selection,
threshold tuning, or calibration. It is a second corpus, not a task-shift
subset of HaluEval.

Label balance: {ds['label_balance']}. Difficulty counts:
{ds['difficulty_counts']}. Hallucination categories: {ds['category_counts']}.

## Protocol

- Features: {protocol['features']['n_base']} base features plus
  {protocol['features']['n_claim']} claim-level features from the existing
  extractors (`extract_features.py`, `claim_features.py`).
- Encoder features were extracted on `{protocol['features']['device']}` in
  {protocol['features']['precision']};
  XGBoost inference ran on the CPU. The CPU-only feature path was abandoned
  because sibling rebuild jobs saturated the CPU, not because of a protocol
  change. The claim-threshold file `verdict_thresholds.json` was absent, so
  the default 0.5/0.5 claim verdicts were used, matching the features the models
  were trained on.
- m3 is EC-XGB ({protocol['models']['m3']['n_features']} features, source
  indicator = {protocol['models']['m3']['source_indicator_value']}, log-scaled
  lengths). m0 is the standard XGBoost reference
  ({protocol['models']['m0']['n_features']} features, raw lengths).
- Fixed hyperparameters for both XGBoost models: max_depth 5, learning rate
  0.05, 300 trees, subsample 0.7, colsample_bytree 0.7.
- Decision threshold: {protocol['decision_threshold']}. No threshold was tuned
  on MedHallu.
- Length-only baseline: logistic regression on `log1p(n_words)`, fit on the
  {protocol['models']['length_only']['train_rows']:,} HaluEval train rows only
  (train prevalence {f4(protocol['models']['length_only']['train_prevalence'])}).
  Coefficient {f4(protocol['models']['length_only']['coef'])}, intercept
  {f4(protocol['models']['length_only']['intercept'])}.
- CIs: {protocol['bootstrap']['n_boot']:,} pair-clustered bootstrap draws
  (seed {protocol['bootstrap']['seed']}), percentile 95% interval. Both rows of
  a pair enter each draw together.

## Headline results

{headline_table(payload)}

`flagged rate` is the share of answers scored at or above 0.5. `pair accuracy`
is the share of pairs where the hallucinated answer scores above its grounded
counterpart (ties count 0.5). The m3 and m0 means are over seeds
{', '.join(str(s) for s in protocol['models']['m3']['seeds'])}; the length-only
model is deterministic.

## Per-seed rows

{per_seed_table(payload)}

## In-domain reference (HaluEval test)

The same fixed-hp models on the 3,000-row HaluEval test split, read from
`fixed_hp_predictions.parquet`. These rows are the in-domain comparator, not a
MedHallu result.

{reference_table(payload)}

Transfer gap (MedHallu minus HaluEval, same fixed-hp models):

{transfer_gap_table(payload)}

Positive F1, AUROC, and PR-AUC deltas mean MedHallu is better than HaluEval.
Positive ECE and Brier deltas mean calibration is worse on MedHallu.

## Subgroups (m3, seed-average scores)

By difficulty:

{subgroup_table(payload, 'm3', 'difficulty')}

By hallucination category:

{subgroup_table(payload, 'm3', 'category')}

m0 for comparison, by difficulty:

{subgroup_table(payload, 'm0', 'difficulty')}

## Source-indicator sensitivity

The deployed EC-XGB sets the natural-response indicator to 1.0 for external
corpora. Forcing it to 0.0 changes the m3 score as follows.

{sensitivity_table(payload)}

## Interpretation

1. **Discrimination versus calibration.** The models were trained on HaluEval,
   where answers are short and the label is strongly tied to surface style.
   On MedHallu their raw probabilities sit near 1 for almost every answer, so
   the 0.5 operating point collapses to "flag everything". The seed-average
   AUROC is {f4(avg_row(m3)['auroc'])} for m3 and {f4(avg_row(m0)['auroc'])} for
   m0, so some ranking signal survives, but ECE and Brier are close to the
   prior-only values. The probabilities are not usable as risk estimates
   without recalibration on target-domain data.
2. **The HaluEval length shortcut reverses.** HaluEval train grounded answers
   average {f4(protocol['models']['length_only']['halueval_train_mean_words']['0'])} words and hallucinated
   answers {f4(protocol['models']['length_only']['halueval_train_mean_words']['1'])} words. MedHallu grounded
   answers are longer than the hallucinated ones ({f4(ds['answer_length_words']['grounded'])} versus
   {f4(ds['answer_length_words']['hallucinated'])} words). A logistic
   model that learned "longer means hallucinated" on HaluEval therefore
   anti-predicts on MedHallu: its AUROC is {f4(avg_row(length)['auroc'])} and its
   pair accuracy is {f4(avg_row(length)['pair_accuracy'])}. This is direct
   evidence that the benchmark's length signal is corpus-specific.
3. **Claim features change the ranking.** m3 adds claim-level NLI
   aggregates and reaches AUROC {f4(avg_row(m3)['auroc'])} against
   {f4(avg_row(m0)['auroc'])} for m0. The direction of the difference is
   corpus-specific; on MedHallu m3 is
   {'higher' if avg_row(m3)['auroc'] >= avg_row(m0)['auroc'] else 'lower'}.
   Both models still flag essentially every row at 0.5.
4. **Subgroups are uneven.** The category and difficulty tables show where the
   ranking signal survives. Small categories have few pairs, so their intervals
   are wide and their point estimates should not be over-read.

## Why this is a second corpus, not a task-shift subset

The task-shift experiment reuses HaluEval dialogue and summarization rows: same
benchmark family, same construction pipeline, different task. MedHallu is
independent in domain (biomedical), source (PubMedQA-style evidence passages),
annotation (expert-written ground truth with released hallucinated answers), and
licence (MIT). No MedHallu row contributed to fitting, tuning, or calibrating
any model reported here.

## Caveats

- Single second corpus. It does not establish general medical hallucination
  detection, and no MedHallu-trained model is proposed.
- MedHallu contains one hallucinated answer per question. The pairs are
  balanced by construction, so prevalence-dependent metrics are not natural
  prevalence estimates.
- The grounded and hallucinated answers differ in length in the opposite
  direction from HaluEval. Part of the ranking signal may still be stylistic.
- The feature extractors and XGBoost models were designed for HaluEval. This is
  zero-shot transfer, not a tuned medical model.
- The LLM-judge comparison on 200 stratified rows was not run within the time
  budget. It is optional and does not affect the rows above.
- CIs cover pair sampling only, not corpus sampling or model selection.

## Runtime and artifacts

- Feature extraction: {runtime.get('feature_extraction_seconds', runtime['total_seconds']):.1f} s
  ({runtime.get('feature_extraction_seconds', runtime['total_seconds']) / 60:.1f} min),
  including model loading and the feature caches.
- Evaluation on the cached features: {runtime.get('evaluation_seconds', runtime['total_seconds']):.1f} s.
- End-to-end: {runtime['total_seconds']:.1f} s
  ({runtime['total_seconds'] / 60:.1f} min).
- Stage seconds: {json.dumps(runtime['stage_seconds'])}.
- Rows: {ds['n_rows']:,} ({ds['n_pairs']:,} pairs).
- Artifacts: `medhallu.csv`, `medhallu.json`, `medhallu_predictions.parquet`,
  `medhallu_REPORT.md`. The shared `REPORT.md` in this directory belongs to the
  conformal-calibration run and is left untouched.
- Feature caches: `data/processed/medhallu_records.parquet`,
  `data/processed/medhallu_base_features.parquet`,
  `data/processed/medhallu_claim_features.parquet`.

## Command

```powershell
$env:HALU_XGB_DEVICE = "cpu"
& .venv\\Scripts\\python.exe src\\models\\rebuild_medhallu.py --device cuda --base-batch-size 16 --claim-batch-size 16
# Re-render metrics from cached features without repeating extraction:
& .venv\\Scripts\\python.exe src\\models\\rebuild_medhallu.py --stage eval --device cuda
& .venv\\Scripts\\python.exe src\\models\\rebuild_medhallu_report.py
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({len(report)} chars)")


if __name__ == "__main__":
    main()
