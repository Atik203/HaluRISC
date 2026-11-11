"""Audit the rebuilt paper numbers against the rebuild artifacts.

This complements `review_number_audit.py` (which covers the unchanged
pre-rebuild values). Every check reads a frozen artifact; no number is entered
without a source. Run from the repo root:

  python src/models/rebuild_number_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "results" / "rebuild"
OUT = RES / "review"
rows = []


def check(name, paper, actual, tol=0.0006):
    paper = float(paper)
    actual = float(actual)
    diff = abs(paper - actual)
    rows.append({
        "check": name,
        "paper_value": paper,
        "artifact_value": actual,
        "abs_diff": diff,
        "status": "match" if diff <= tol else "MISMATCH",
    })


fixed = pd.read_csv(RES / "fixed_hp" / "fixed_hp_ablation.csv")
fixed_mean = fixed.groupby("variant")[["f1", "auroc", "ece", "brier", "mcc"]].mean()
ext = pd.read_csv(RES / "fixed_hp" / "b6_external_metrics.csv")
ext_mean = ext.groupby(["variant", "dataset"])[["f1", "auroc", "ece"]].mean()

for variant, f1, auc, mcc, ece in (
    ("m0", 0.9849, 0.9979, 0.9701, 0.0050),
    ("m1", 0.9853, 0.9976, 0.9710, 0.0042),
    ("m2", 0.9849, 0.9985, 0.9701, 0.0051),
    ("m3", 0.9854, 0.9986, 0.9713, 0.0078),
):
    check(f"Table 6 {variant} F1", f1, fixed_mean.loc[variant, "f1"])
    check(f"Table 6 {variant} AUROC", auc, fixed_mean.loc[variant, "auroc"])
    check(f"Table 6 {variant} MCC", mcc, fixed_mean.loc[variant, "mcc"])
    check(f"Table 6 {variant} ECE", ece, fixed_mean.loc[variant, "ece"])

for variant, auc, ece in (("m0", 0.4937, 0.6352), ("m1", 0.5567, 0.6260), ("m2", 0.5461, 0.6359), ("m3", 0.5898, 0.2746)):
    check(f"Table 6 {variant} RAGTruth AUROC", auc, ext_mean.loc[(variant, "ragtruth_all_test"), "auroc"])
    check(f"Table 6 {variant} RAGTruth ECE", ece, ext_mean.loc[(variant, "ragtruth_all_test"), "ece"])

# WP2 calibration
cal = pd.read_csv(RES / "calibration" / "calibration.csv")
qa = cal[(cal["dataset"] == "ragtruth_qa_test") & (cal["seed"].astype(str) == "pooled") & (cal["model"] == "fixed_hp")].set_index("calibrator")
for name, ece, brier, bss in (
    ("raw", 0.731, 0.689, -3.71),
    ("source_platt", 0.765, 0.742, -4.07),
    ("target_platt", 0.125, 0.161, -0.10),
):
    check(f"Table 8 {name} ECE", ece, qa.loc[name, "ece"])
    check(f"Table 8 {name} Brier", brier, qa.loc[name, "brier"])
    check(f"Table 8 {name} BSS", bss, qa.loc[name, "brier_skill_score"], tol=0.006)

check("Table 8 target AUROC", 0.558, qa.loc["target_platt", "auroc"])
check("Table 8 target flagged", 0.0, 100 * qa.loc["target_platt", "predicted_positive_rate"], tol=0.06)
check("Table 8 raw flagged", 96.8, 100 * qa.loc["raw", "predicted_positive_rate"], tol=0.06)

slope = pd.read_csv(RES / "calibration" / "calibration_slope.csv")
slope = slope[(slope["model"] == "fixed_hp") & (slope["fit_split"] == "ragtruth_qa_calibration")]
check("Table 8 target slope", 0.18, slope["cal_slope"].mean())
check("Table 8 target intercept", -1.419, slope["cal_intercept"].mean(), tol=0.006)

prior = pd.read_csv(RES / "calibration" / "prior_only.csv")
prior_test = prior[(prior["predictor"] == "test_prior") & (prior["eval_split"] == "ragtruth_qa_test")].iloc[0]
prior_cal = prior[(prior["predictor"] == "calibration_prior") & (prior["eval_split"] == "ragtruth_qa_test")].iloc[0]
check("Table 8 prior test Brier", 0.146, prior_test["brier"])
check("Table 8 prior cal ECE", 0.133, prior_cal["ece"])
check("Table 8 prior cal Brier", 0.164, prior_cal["brier"])

budget = pd.read_csv(RES / "calibration" / "label_budget.csv")
budget = budget[budget["seed"].astype(str) == "pooled"].set_index("size")
check("Figure 9 50-row ECE mean", 0.130, budget.loc[50, "ece_mean"])
check("Figure 9 50-row ECE low", 0.020, budget.loc[50, "ece_q025"])
check("Figure 9 50-row ECE high", 0.223, budget.loc[50, "ece_q975"])
check("Figure 9 252-row ECE mean", 0.124, budget.loc[252, "ece_mean"])
check("Figure 9 252-row ECE low", 0.075, budget.loc[252, "ece_q025"])
check("Figure 9 252-row ECE high", 0.188, budget.loc[252, "ece_q975"])
check("Figure 9 full ECE", 0.125, budget.loc[5034, "ece_mean"])

# WP3 intervals
inter = pd.read_csv(RES / "intervals" / "intervals.csv")
inter = inter[(inter["source"] == "fixed_hp") & (inter["seed"].astype(str) == "pooled")]

def interval(dataset, comparison, metric):
    r = inter[(inter["dataset"] == dataset) & (inter["comparison"] == comparison) & (inter["metric"] == metric)].iloc[0]
    return r["point_delta"], r["ci_low"], r["ci_high"]

for dataset, comp, metric, point, lo, hi in (
    ("halueval_test", "m3-m0", "f1", 0.0006, -0.0010, 0.0021),
    ("halueval_test", "m3-m0", "auroc", 0.0007, 0.0002, 0.0012),
    ("halueval_test", "m2-m1", "f1", -0.0004, -0.0020, 0.0012),
    ("halueval_test", "m3-m1", "f1", 0.0001, -0.0015, 0.0018),
    ("ragtruth_qa_test", "m3-m0", "auroc", 0.0017, -0.0270, 0.0291),
    ("ragtruth_all_test", "m3-m0", "f1", 0.0451, 0.0348, 0.0554),
):
    p, l, h = interval(dataset, comp, metric)
    check(f"Table 11 {dataset} {comp} {metric} point", point, p)
    check(f"Table 11 {dataset} {comp} {metric} low", lo, l)
    check(f"Table 11 {dataset} {comp} {metric} high", hi, h)

# WP4 explanations
shap = pd.read_csv(RES / "explanations" / "m3_global_shap_importance.csv")
shap = shap.set_index("feature")["mean_abs_shap"]
for feature, value in (("overlap_answer_context", 1.66), ("n_words", 1.11), ("cosine_q_ans", 0.96), ("n_chars", 0.82), ("jaccard_ans_q", 0.78), ("claim_mean_contradicts", 0.22)):
    check(f"Figure 10 {feature}", value, shap.loc[feature], tol=0.006)

comp = json.loads((RES / "explanations" / "m3_vs_b2_comparison.json").read_text())
check("Figure 10 Spearman", 0.878, comp["spearman"], tol=0.001)
check("Figure 10 Kendall", 0.730, comp["kendall_tau"], tol=0.001)

pert = pd.read_csv(RES / "explanations" / "m3_perturbation_aggregates.csv").set_index("perturbation")
for operator, delta, flip in (("entity", 0.339, 0.545), ("support_removal", 0.464, 0.0), ("numeric", 0.332, 0.0), ("date", 0.482, 0.0), ("irrelevant_insert", 0.003, 0.0), ("clause_shuffle", 0.0001, 0.0)):
    check(f"Table 12 {operator} delta", delta, pert.loc[operator, "mean_abs_score_delta"], tol=0.0006)
    check(f"Table 12 {operator} flip", flip, pert.loc[operator, "top1_flip_rate"], tol=0.0006)

# WP5 audit counts
err = pd.read_csv(ROOT / "docs" / "rebuild" / "wp5_error_labels.csv")
check("WP5 error rows", 20, len(err))
check("WP5 error content", 13, (err["primary_category"].fillna("") != "").sum() - 7)
check("WP5 error artifacts", 7, err["primary_category"].str.startswith("artifact_").sum())
exp = pd.read_csv(ROOT / "docs" / "rebuild" / "wp5_explanation_labels.csv")
check("WP5 explanation implausible", 13, (exp["explanation_plausible"] == "implausible").sum())
check("WP5 explanation unsure", 3, (exp["explanation_plausible"] == "unsure").sum())
check("WP5 explanation plausible", 10, (exp["explanation_plausible"] == "plausible").sum())

# WP6 judge and latency
judge = json.loads((RES / "judge" / "metrics_openai_full.json").read_text())
mean = judge["mean_across_seeds"]
check("Text GPT accuracy", 0.743, mean["accuracy"]["mean"])
check("Table 15 GPT precision", 0.404, mean["precision"]["mean"])
check("Table 15 GPT recall", 0.942, mean["recall"]["mean"])
check("Table 15 GPT F1", 0.565, mean["f1"]["mean"])
check("Table 15 GPT AUROC", 0.910, mean["auroc"]["mean"])
check("Table 15 GPT flagged", 41.4, 100 * mean["predicted_positive_rate"]["mean"], tol=0.06)
check("Table 15 GPT p50", 2283, judge["latency"]["p50_ms"], tol=0.6)
check("Table 15 GPT cost/1K", 0.234, judge["cost"]["usd_per_1000_calls"], tol=0.0006)

qwen = json.loads((RES / "judge" / "metrics_qwen_full.json").read_text())
qmean = qwen["mean_across_seeds"]
check("Text Qwen accuracy", 0.850, qmean["accuracy"]["mean"])
check("Table 15 Qwen precision", 0.570, qmean["precision"]["mean"])
check("Table 15 Qwen recall", 0.638, qmean["recall"]["mean"])
check("Table 15 Qwen F1", 0.602, qmean["f1"]["mean"])
check("Table 15 Qwen AUROC", 0.778, qmean["auroc"]["mean"])
check("Table 15 Qwen flagged", 19.9, 100 * qmean["predicted_positive_rate"]["mean"], tol=0.06)
check("Table 15 Qwen p50", 3050, qwen["latency"]["p50_ms"], tol=0.6)

lat = json.loads((RES / "latency" / "m3_latency.json").read_text())
check("Table 13 m3 total p50", 127.5, lat["total"]["p50_ms"], tol=0.06)
check("Table 13 m3 total p95", 234.6, lat["total"]["p95_ms"], tol=0.06)
check("Table 13 claim p50", 43.8, lat["claim"]["p50_ms"], tol=0.06)
check("Table 14 encoder p50", 6.97, json.loads((RES / "judge" / "encoder_timing.json").read_text())["ms_per_row"], tol=0.006)

# WP7 length audit
length = json.loads((RES / "length_control_fixed_hp.json").read_text())
check("Length control matched AUROC", 0.956, length["matched"]["ec_xgb"]["mean"]["auroc"], tol=0.0006)
check("Length control matched length-only AUROC", 0.521, length["matched"]["length_only"]["auroc"], tol=0.0006)
check("Length control matched AUROC gap", 0.436, length["matched"]["delta_ec_xgb_minus_length_only"]["auroc_delta"], tol=0.0006)
check("Length control matched gap low", 0.351, length["matched"]["delta_ec_xgb_minus_length_only"]["auroc_ci95"][0], tol=0.0006)
check("Length control matched gap high", 0.522, length["matched"]["delta_ec_xgb_minus_length_only"]["auroc_ci95"][1], tol=0.0006)

frame = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
frame.to_csv(OUT / "rebuild_number_audit.csv", index=False)
(OUT / "rebuild_number_audit.json").write_text(
    json.dumps({
        "n_checks": len(frame),
        "n_match": int((frame["status"] == "match").sum()),
        "n_mismatch": int((frame["status"] == "MISMATCH").sum()),
    }, indent=2),
    encoding="utf-8",
)
print(json.dumps({
    "n_checks": len(frame),
    "n_match": int((frame["status"] == "match").sum()),
    "n_mismatch": int((frame["status"] == "MISMATCH").sum()),
}, indent=2))
if (frame["status"] == "MISMATCH").any():
    print(frame[frame["status"] == "MISMATCH"].to_string(index=False))
