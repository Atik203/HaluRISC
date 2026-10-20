"""Trace every paper number to a frozen artifact (P1, WS-D2).

The audit is intentionally explicit: each row records the paper location, the
printed value, the artifact path, the artifact key, and whether the value
matches within the printed precision. Run from the repository root:

  python src/models/review_number_audit.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "results"
OUT = RES / "review"

rows = []


def check(paper, expected, actual, artifact, key, tol=0.0006):
    expected = float(expected)
    actual = float(actual)
    status = "match" if abs(expected - actual) <= tol else "MISMATCH"
    rows.append({
        "paper": paper,
        "paper_value": expected,
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_key": key,
        "artifact_value": actual,
        "abs_diff": abs(expected - actual),
        "status": status,
    })


b2 = pd.read_csv(RES / "b2" / "b2_model_comparison.csv").set_index("model")
b6 = pd.read_csv(RES / "b6" / "b6_model_comparison.csv").set_index("Unnamed: 0")
ext = pd.read_csv(RES / "b6" / "b6_external_metrics.csv")
b4 = json.loads((RES / "b4" / "b4_calibration_metrics.json").read_text())
cal = json.loads((RES / "review" / "calibration_protocol.json").read_text())
extras = json.loads((RES / "extras" / "review_extras.json").read_text())
abl = pd.read_csv(RES / "ablation_results.csv").set_index("removed_group")
pert = pd.read_csv(RES / "b5" / "b5_perturbation_aggregates.csv").set_index("perturbation")
lat = json.loads((RES / "latency_analysis.json").read_text())
judge = json.loads((RES / "llm_judge_results.json").read_text())
gpt_new = json.loads((RES / "review" / "llm_judge_gpt_ec_xgb.json").read_text())
qwen_new = json.loads((RES / "review" / "llm_judge_qwen.json").read_text())
err = json.loads((RES / "error_analysis.json").read_text())
length = pd.read_csv(RES / "b5" / "b5_length_error_analysis.csv").set_index("bin")

# Table 4: model comparison (HaluEval test, three seeds)
t4 = {
    "xgboost": ("0.9846", "0.9979"),
    "rf_full": ("0.9842", "0.9980"),
    "lr_full": ("0.9660", "0.9932"),
    "heuristic_overlap": ("0.9352", "0.9132"),
    "nli_only": ("0.6714", "0.7177"),
    "tfidf_answer": ("0.9224", "0.9696"),
    "tfidf_all": ("0.5999", "0.6754"),
    "majority": ("0.0000", "0.0000"),
}
for model, (f1, auc) in t4.items():
    check(f"Table 4 / {model} / F1", f1, b2.loc[model, "f1_mean"], RES / "b2" / "b2_model_comparison.csv", f"{model}.f1_mean")
    if model != "majority":
        check(f"Table 4 / {model} / AUROC", auc, b2.loc[model, "auroc_mean"], RES / "b2" / "b2_model_comparison.csv", f"{model}.auroc_mean")

# Table 5: EC-XGB component ablation
t5 = {
    "m0": ("0.9844", "0.9980", "0.9693", "0.0066"),
    "m1": ("0.9852", "0.9976", "0.9708", "0.0051"),
    "m2": ("0.9857", "0.9984", "0.9717", "0.0078"),
    "m3": ("0.9855", "0.9984", "0.9715", "0.0074"),
}
for variant, (f1, auc, mcc, ece) in t5.items():
    for metric, expected in (("f1_mean", f1), ("auroc_mean", auc), ("mcc_mean", mcc), ("ece_mean", ece)):
        check(f"Table 5 / {variant} / {metric}", expected, b6.loc[variant, metric], RES / "b6" / "b6_model_comparison.csv", f"{variant}.{metric}")

# Table 6: cross-domain behavior
t6 = {
    ("m0", "ragtruth_all_test"): ("0.5181", "0.4753", "0.9989", "0.6346"),
    ("m0", "ragtruth_qa_test"): ("0.3022", "0.5465", "0.9989", "0.8149"),
    ("m0", "faithbench"): ("0.8130", "0.5089", "0.9947", "0.2917"),
    ("m3", "ragtruth_all_test"): ("0.5608", "0.5815", "0.6132", "0.2782"),
    ("m3", "ragtruth_qa_test"): ("0.3081", "0.5491", "0.9741", "0.7433"),
    ("m3", "faithbench"): ("0.1201", "0.5941", "0.0733", "0.4500"),
}
for (variant, dataset), (f1, auc, flag, ece) in t6.items():
    row = ext[(ext["variant"] == variant) & (ext["dataset"] == dataset)].iloc[0]
    for metric, expected in (("f1", f1), ("auroc", auc), ("predicted_positive_rate", flag), ("ece", ece)):
        check(f"Table 6 / {variant} / {dataset} / {metric}", expected, row[metric], RES / "b6" / "b6_external_metrics.csv", f"{variant}.{dataset}.{metric}")

# Table 7: HaluEval calibration
t7 = {
    "raw": ("0.9846", "0.0045", "0.0117", "0.0449"),
    "platt": ("0.9848", "0.0091", "0.0129", "0.0579"),
    "isotonic": ("0.9844", "0.0070", "0.0124", "0.0727"),
}
for method, (f1, ece, brier, nll) in t7.items():
    block = b4["halueval_test"][method]
    for key, expected in (("f1_mean", f1), ("ece_mean", ece), ("brier_mean", brier), ("nll_mean", nll)):
        check(f"Table 7 / {method} / {key}", expected, block[key], RES / "b4" / "b4_calibration_metrics.json", f"halueval_test.{method}.{key}")

# Table 8: combined calibration
t8 = {
    ("standard", "raw"): (0.8185, 0.8164, 5.20, 0.999),
    ("standard", "source_platt"): (0.8089, 0.8011, 3.62, 0.999),
    ("standard", "target_platt"): (0.1335, 0.1639, 0.51, 0.000),
    ("standard", "target_isotonic"): (0.1316, 0.1663, 0.53, 0.000),
    ("ec_xgb", "raw"): (0.7433, 0.7061, 2.91, 0.974),
    ("ec_xgb", "source_platt"): (0.7735, 0.7527, 3.18, 0.975),
    ("ec_xgb", "target_platt"): (0.1263, 0.1613, 0.50, 0.000),
    ("ec_xgb", "target_isotonic"): (0.1260, 0.1618, 0.50, 0.000),
}
for (model_name, method), values in t8.items():
    block = cal["standard"][method] if model_name == "standard" else cal["ec_xgb"]["mean"][method]
    for metric, expected in zip(("ece", "brier", "nll", "predicted_positive_rate"), values):
        check(f"Table 8 / {model_name} / {method} / {metric}", expected, block[metric], RES / "review" / "calibration_protocol.json", f"{model_name}.{method}.{metric}", tol=0.006)

# Table 9: feature-group ablation
t9 = {
    "length": ("0.9801", "0.9963"),
    "lexical": ("0.9600", "0.9923"),
    "entity": ("0.9847", "0.9982"),
    "nli": ("0.9823", "0.9973"),
    "numeric": ("0.9844", "0.9982"),
    "hedging": ("0.9848", "0.9981"),
    "semantic": ("0.9860", "0.9982"),
}
for group, (f1, auc) in t9.items():
    check(f"Table 9 / {group} / F1", f1, abl.loc[group, "f1_mean"], RES / "ablation_results.csv", f"{group}.f1_mean")
    check(f"Table 9 / {group} / AUROC", auc, abl.loc[group, "auroc_mean"], RES / "ablation_results.csv", f"{group}.auroc_mean")

# Table 10: perturbation study
t10 = {
    "entity": ("0.360", "0.840"),
    "numeric": ("0.345", "0.852"),
    "date": ("0.279", "0.870"),
    "support_removal": ("0.272", "0.874"),
    "irrelevant_insert": ("0.006", "0.948"),
    "clause_shuffle": ("0.001", "0.921"),
}
for op, (delta, rho) in t10.items():
    check(f"Table 10 / {op} / mean delta", delta, pert.loc[op, "mean_abs_score_delta"], RES / "b5" / "b5_perturbation_aggregates.csv", f"{op}.mean_abs_score_delta", tol=0.001)
    check(f"Table 10 / {op} / Spearman", rho, pert.loc[op, "mean_spearman"], RES / "b5" / "b5_perturbation_aggregates.csv", f"{op}.mean_spearman", tol=0.001)

# Table 11: latency
t11 = {
    "core_lexical": 0.1,
    "entity": 21.13,
    "nli": 27.09,
    "semantic": 9.25,
    "model": 1.26,
    "shap": 1.91,
}
for stage, expected in t11.items():
    check(f"Table 11 / {stage} / p50", expected, lat["latency_ms"][stage]["p50"], RES / "latency_analysis.json", f"latency_ms.{stage}.p50", tol=0.011)
check("Table 11 / total / p50", 61.8, lat["total_per_sample_ms"]["p50"], RES / "latency_analysis.json", "total_per_sample_ms.p50", tol=0.05)

# Table 12: LLM judge
for key, expected in (("accuracy", 0.865), ("precision", 0.987), ("recall", 0.740), ("f1", 0.846)):
    check(f"Table 12 / GPT / {key}", expected, gpt_new["gpt"][key],
          RES / "review" / "llm_judge_gpt_ec_xgb.json", f"gpt.{key}")
for key, expected in (("accuracy", 0.800), ("precision", 0.969), ("recall", 0.620), ("f1", 0.756)):
    check(f"Table 12 / Qwen / {key}", expected, qwen_new["qwen"][key],
          RES / "review" / "llm_judge_qwen.json", f"qwen.{key}")
for key, expected in (("accuracy", 0.990), ("precision", 0.990), ("recall", 0.990), ("f1", 0.990)):
    check(f"Table 12 / EC-XGB / {key}", expected, gpt_new["ec_xgb_same_rows"][key],
          RES / "review" / "llm_judge_gpt_ec_xgb.json", f"ec_xgb_same_rows.{key}")
check("Table 12 / GPT / p50 ms", 1443, gpt_new["latency_ms_p50"],
      RES / "review" / "llm_judge_gpt_ec_xgb.json", "latency_ms_p50", tol=2)
check("Table 12 / GPT / cost per 1k", 0.106, gpt_new["cost_per_1000_usd"],
      RES / "review" / "llm_judge_gpt_ec_xgb.json", "cost_per_1000_usd", tol=0.001)
check("Table 12 / Qwen / p50 ms", 2623, qwen_new["latency_ms_p50"],
      RES / "review" / "llm_judge_qwen.json", "latency_ms_p50", tol=2)

# Table 13: error taxonomy
check("Table 13 / label ambiguity FP", 7, err["category_counts"]["false_positive"]["label_ambiguity"], RES / "error_analysis.json", "category_counts.false_positive.label_ambiguity", tol=0)
check("Table 13 / label ambiguity FN", 4, err["category_counts"]["false_negative"]["label_ambiguity"], RES / "error_analysis.json", "category_counts.false_negative.label_ambiguity", tol=0)
check("Table 13 / short-answer FP", 3, err["category_counts"]["false_positive"]["short_answer_ambiguity"], RES / "error_analysis.json", "category_counts.false_positive.short_answer_ambiguity", tol=0)
check("Table 13 / short-answer FN", 6, err["category_counts"]["false_negative"]["short_answer_ambiguity"], RES / "error_analysis.json", "category_counts.false_negative.short_answer_ambiguity", tol=0)

# Table 14: length analysis
t14 = {
    "1": ("524", "0.021", "0.024"),
    "2": ("514", "0.033", "0.035"),
    "3-4": ("474", "0.127", "0.126"),
    "5-8": ("641", "0.902", "0.888"),
    "9-16": ("600", "0.982", "0.980"),
    "17+": ("247", "0.992", "0.993"),
}
for bin_name, (n, pos, score) in t14.items():
    check(f"Table 14 / {bin_name} / rows", n, length.loc[bin_name, "rows"], RES / "b5" / "b5_length_error_analysis.csv", f"{bin_name}.rows", tol=0)
    check(f"Table 14 / {bin_name} / pos rate", pos, length.loc[bin_name, "pos_rate"], RES / "b5" / "b5_length_error_analysis.csv", f"{bin_name}.pos_rate", tol=0.001)
    check(f"Table 14 / {bin_name} / mean score", score, length.loc[bin_name, "mean_score"], RES / "b5" / "b5_length_error_analysis.csv", f"{bin_name}.mean_score", tol=0.001)

# Headline numbers and statistics
check("Headline / EC-XGB HaluEval F1", 0.9855, b6.loc["m3", "f1_mean"], RES / "b6" / "b6_model_comparison.csv", "m3.f1_mean")
check("Headline / EC-XGB HaluEval AUROC", 0.9984, b6.loc["m3", "auroc_mean"], RES / "b6" / "b6_model_comparison.csv", "m3.auroc_mean")
check("Headline / RAGTruth all flag standard", 0.999, ext[(ext.variant == "m0") & (ext.dataset == "ragtruth_all_test")].iloc[0]["predicted_positive_rate"], RES / "b6" / "b6_external_metrics.csv", "m0.ragtruth_all_test.predicted_positive_rate")
check("Headline / RAGTruth all flag EC-XGB", 0.613, ext[(ext.variant == "m3") & (ext.dataset == "ragtruth_all_test")].iloc[0]["predicted_positive_rate"], RES / "b6" / "b6_external_metrics.csv", "m3.ragtruth_all_test.predicted_positive_rate")
check("Headline / RAGTruth all AUROC standard", 0.475, ext[(ext.variant == "m0") & (ext.dataset == "ragtruth_all_test")].iloc[0]["auroc"], RES / "b6" / "b6_external_metrics.csv", "m0.ragtruth_all_test.auroc")
check("Headline / RAGTruth all AUROC EC-XGB", 0.582, ext[(ext.variant == "m3") & (ext.dataset == "ragtruth_all_test")].iloc[0]["auroc"], RES / "b6" / "b6_external_metrics.csv", "m3.ragtruth_all_test.auroc")
check("Headline / held-out QA F1", 0.30, ext[(ext.variant == "m3") & (ext.dataset == "ragtruth_qa_test")].iloc[0]["f1"], RES / "b6" / "b6_external_metrics.csv", "m3.ragtruth_qa_test.f1", tol=0.01)
check("Headline / EC-XGB display ECE before", 0.726, cal["deployed_display"]["test_raw"]["ece"], RES / "b6" / "b6_calibration.json", "test_raw.ece", tol=0.001)
check("Headline / EC-XGB display ECE after", 0.126, cal["deployed_display"]["test_platt"]["ece"], RES / "b6" / "b6_calibration.json", "test_platt.ece", tol=0.001)
check("Headline / target prevalence", 0.178, cal["meta"]["positive_prevalence_test"], RES / "review" / "calibration_protocol.json", "meta.positive_prevalence_test", tol=0.001)
check("Headline / calibration rows", 5034, cal["meta"]["n_calibration"], RES / "review" / "calibration_protocol.json", "meta.n_calibration", tol=0)
check("Headline / test rows", 900, cal["meta"]["n_test"], RES / "review" / "calibration_protocol.json", "meta.n_test", tol=0)
check("Headline / 252-row ECE", 0.133, extras["x4_label_budget"]["0.05"]["ece"], RES / "extras" / "review_extras.json", "x4_label_budget.0.05.ece", tol=0.001)
check("Headline / McNemar XGB vs RF", 0.044, json.loads((RES / "b2" / "b2_statistical_tests.json").read_text())["mcnemar_pairs"]["rf_full"], RES / "b2" / "b2_statistical_tests.json", "mcnemar_pairs.rf_full", tol=0.001)
check("Headline / McNemar m0 vs m3", 0.45, json.loads((RES / "b6" / "b6_statistical_tests.json").read_text())["mcnemar"]["m0_vs_m3"], RES / "b6" / "b6_statistical_tests.json", "mcnemar.m0_vs_m3", tol=0.01)
check("Headline / Wilcoxon m0 vs m3", 0.25, json.loads((RES / "b6" / "b6_statistical_tests.json").read_text())["wilcoxon"]["m0_vs_m3"], RES / "b6" / "b6_statistical_tests.json", "wilcoxon.m0_vs_m3", tol=0.01)
check("Headline / judge p-value", 8.32e-6, gpt_new["mcnemar_p"], RES / "review" / "llm_judge_gpt_ec_xgb.json", "mcnemar_p", tol=1e-7)
check("Headline / judge discordant 27", 27, gpt_new["discordant_pairs"]["gpt_wrong_ec_right"], RES / "review" / "llm_judge_gpt_ec_xgb.json", "discordant_pairs.gpt_wrong_ec_right", tol=0)
check("Headline / judge discordant 2", 2, gpt_new["discordant_pairs"]["gpt_right_ec_wrong"], RES / "review" / "llm_judge_gpt_ec_xgb.json", "discordant_pairs.gpt_right_ec_wrong", tol=0)

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "number_audit.csv", index=False)

summary = {
    "n_checks": int(len(df)),
    "n_match": int((df["status"] == "match").sum()),
    "n_mismatch": int((df["status"] == "MISMATCH").sum()),
    "mismatches": df[df["status"] == "MISMATCH"].to_dict(orient="records"),
}
(OUT / "number_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("n_checks", "n_match", "n_mismatch")}, indent=2))
if summary["n_mismatch"]:
    print(df[df["status"] == "MISMATCH"].to_string(index=False))
