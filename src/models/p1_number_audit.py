"""Audit the P0/P1 paper numbers against the frozen artifacts.

Every check reads an artifact under artifacts/results/rebuild/p0 or p1. The
`paper` value is the number printed in the manuscript. A mismatch means either
the artifact or the manuscript changed. Run from the repo root:

  python src/models/p1_number_audit.py

Writes artifacts/results/rebuild/review/p1_number_audit.csv and prints a JSON
summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P0 = ROOT / "artifacts" / "results" / "rebuild" / "p0"
P1 = ROOT / "artifacts" / "results" / "rebuild" / "p1"
OUT = ROOT / "artifacts" / "results" / "rebuild" / "review"
OUT.mkdir(parents=True, exist_ok=True)

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


def mean_json(values):
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# P0.2 feature-group ablation
# ---------------------------------------------------------------------------
abl = pd.read_csv(P0 / "feature_group_ablation.csv").set_index("config")
for name, f1, auroc, mcc in (
    ("full_m3", 0.9861, 0.9985, 0.9726),
    ("minus_lexical", 0.9690, 0.9944, 0.9382),
    ("minus_length", 0.9795, 0.9972, 0.9599),
    ("minus_nli", 0.9821, 0.9981, 0.9645),
):
    check(f"Table ablation {name} F1", f1, abl.loc[name, "f1_mean"])
    check(f"Table ablation {name} AUROC", auroc, abl.loc[name, "auroc_mean"])
    check(f"Table ablation {name} MCC", mcc, abl.loc[name, "mcc_mean"])
for name, df1, dauc, dmcc in (
    ("minus_lexical", -0.0171, -0.0040, -0.0343),
    ("minus_length", -0.0066, -0.0012, -0.0127),
    ("minus_nli", -0.0040, -0.0004, -0.0080),
    ("no_claims", -0.0010, -0.0003, -0.0020),
):
    check(f"Table ablation {name} dF1", df1, abl.loc[name, "delta_f1_mean"])
    check(f"Table ablation {name} dAUROC", dauc, abl.loc[name, "delta_auroc_mean"])
    check(f"Table ablation {name} dMCC", dmcc, abl.loc[name, "delta_mcc_mean"])

# ---------------------------------------------------------------------------
# P0.3 SHAP seed stability
# ---------------------------------------------------------------------------
shap = json.loads((P0 / "shap_seed_stability.json").read_text(encoding="utf-8"))
summary = shap["summary"]
ref = shap["reference_vs"]
check("SHAP Spearman seed 123", 0.9717, ref["123"]["spearman"], tol=0.0002)
check("SHAP Spearman seed 456", 0.9501, ref["456"]["spearman"], tol=0.0002)
check("SHAP top10 overlap seed 123", 9, ref["123"]["top10_overlap"])
check("SHAP top10 overlap seed 456", 9, ref["456"]["top10_overlap"])
check("SHAP mean pairwise Spearman", 0.9672, summary["mean_pairwise_spearman"], tol=0.0002)

# ---------------------------------------------------------------------------
# P0 CPU latency
# ---------------------------------------------------------------------------
lat = json.loads((P0 / "m3_latency_cpu.json").read_text(encoding="utf-8"))
check("CPU total mean ms", 2002.15, lat["total"]["mean_ms"], tol=0.8)
check("CPU total p50 ms", 975.36, lat["total"]["p50_ms"], tol=0.8)
check("CPU total p95 ms", 7564.71, lat["total"]["p95_ms"], tol=2.0)

# ---------------------------------------------------------------------------
# P0 judge length-matched
# ---------------------------------------------------------------------------
jm = pd.read_csv(P0 / "judge_length_matched.csv")
gpt = jm[(jm.model_family == "llm_judge") & (jm.model.str.startswith("gpt"))]
ec = jm[jm.model_family == "ec_xgb_m3"]
qw = jm[(jm.model_family == "llm_judge") & (jm.model.str.startswith("qwen"))]
check("Judge length GPT F1", 0.8650, gpt["f1"].mean())
check("Judge length GPT AUROC", 0.9567, gpt["auroc"].mean())
check("Judge length EC F1", 0.9071, ec["f1"].mean())
check("Judge length EC AUROC", 0.9555, ec["auroc"].mean())
check("Judge length Qwen F1", 0.8209, qw["f1"].mean())
check("Judge length Qwen AUROC", 0.8632, qw["auroc"].mean())

# ---------------------------------------------------------------------------
# P0 prompt sensitivity
# ---------------------------------------------------------------------------
ps = json.loads((P0 / "prompt_sensitivity.json").read_text(encoding="utf-8"))
by_prompt = ps["summary"]["gpt"]["by_prompt"]
check("Prompt current F1", 0.8154, by_prompt["current"]["f1"]["mean"])
check("Prompt strict F1", 0.8467, by_prompt["strict_evidence_only"]["f1"]["mean"])
check("Prompt concise F1", 0.8421, by_prompt["concise_json"]["f1"]["mean"])
check("Prompt current AUROC", 0.9197, by_prompt["current"]["auroc"]["mean"])
check("Prompt strict AUROC", 0.9445, by_prompt["strict_evidence_only"]["auroc"]["mean"])
check("Prompt concise AUROC", 0.9420, by_prompt["concise_json"]["auroc"]["mean"])

# ---------------------------------------------------------------------------
# P1 external baseline
# ---------------------------------------------------------------------------
ext = pd.read_csv(P1 / "external_baseline.csv")
ext_lr = ext[ext.model == "external_lr_train"].iloc[0]
ec_m3 = ext[(ext.model == "ec_xgb_m3") & (ext.seed.astype(str) == "pooled")].iloc[0]
check("External LR F1", 0.9517, ext_lr["f1"])
check("External LR AUROC", 0.9850, ext_lr["auroc"])
check("External LR ECE", 0.0141, ext_lr["ece"])
check("External m3 F1", 0.9854, ec_m3["f1"])
check("External m3 AUROC", 0.9986, ec_m3["auroc"])
check("External m3 ECE", 0.0078, ec_m3["ece"])

# ---------------------------------------------------------------------------
# P1 task shift
# ---------------------------------------------------------------------------
ts = pd.read_csv(P1 / "task_shift.csv")
m3 = ts[(ts.model == "m3") & (ts.seed.astype(str) == "mean")].set_index("task")
lenb = ts[(ts.model == "length_logistic")].set_index("task")
m0 = ts[(ts.model == "m0") & (ts.seed.astype(str) == "mean")].set_index("task")
for task, f1, auroc, ece, flag in (
    ("dialogue", 0.6698, 0.7634, 0.4879, 0.9930),
    ("summarization", 0.5831, 0.7167, 0.1133, 0.3750),
):
    check(f"Task shift m3 {task} F1", f1, m3.loc[task, "f1"])
    check(f"Task shift m3 {task} AUROC", auroc, m3.loc[task, "auroc"])
    check(f"Task shift m3 {task} ECE", ece, m3.loc[task, "ece"])
    check(f"Task shift m3 {task} flagged", flag, m3.loc[task, "flagged_rate"])
check("Task shift length dialogue AUROC", 0.7242, lenb.loc["dialogue", "auroc"])
check("Task shift length summarization AUROC", 0.7880, lenb.loc["summarization", "auroc"])
check("Task shift m0 dialogue AUROC", 0.7407, m0.loc["dialogue", "auroc"])
check("Task shift m0 summarization AUROC", 0.6045, m0.loc["summarization", "auroc"])

# ---------------------------------------------------------------------------
# P1 MedHallu
# ---------------------------------------------------------------------------
mh = pd.read_csv(P1 / "medhallu.csv")
mh3 = mh[(mh.model == "m3_seedavg")].iloc[0]
mh0 = mh[(mh.model == "m0_seedavg")].iloc[0]
mhl = mh[(mh.model == "length_only")].iloc[0]
check("MedHallu m3 pair accuracy", 0.8670, mh3["pair_accuracy"])
check("MedHallu m3 AUROC", 0.8212, mh3["auroc"])
check("MedHallu m3 F1", 0.6667, mh3["f1"])
check("MedHallu m3 ECE", 0.4931, mh3["ece"])
check("MedHallu m0 pair accuracy", 0.8220, mh0["pair_accuracy"])
check("MedHallu length pair accuracy", 0.1995, mhl["pair_accuracy"])
check("MedHallu length AUROC", 0.2604, mhl["auroc"])

# ---------------------------------------------------------------------------
# P1 HaluBench
# ---------------------------------------------------------------------------
hb = pd.read_csv(P1 / "halubench.csv")
hb3 = hb[(hb.scope == "overall") & (hb.model == "m3")].iloc[0]
hb0 = hb[(hb.scope == "overall") & (hb.model == "m0")].iloc[0]
hbl = hb[(hb.scope == "overall") & (hb.model == "length_only")].iloc[0]
check("HaluBench m3 F1", 0.5884, hb3["f1"])
check("HaluBench m3 AUROC", 0.6002, hb3["auroc"])
check("HaluBench m3 ECE", 0.3336, hb3["ece"])
check("HaluBench m0 AUROC", 0.6086, hb0["auroc"])
check("HaluBench length AUROC", 0.5143, hbl["auroc"])
for source, auroc in (("DROP", 0.5133), ("FinanceBench", 0.5265), ("covidQA", 0.8325)):
    value = hb[(hb.scope == source) & (hb.model == "m3")].iloc[0]["auroc"]
    check(f"HaluBench m3 {source} AUROC", auroc, value)

# ---------------------------------------------------------------------------
# P1 retrieval stress
# ---------------------------------------------------------------------------
rs = json.loads((P1 / "retrieval_stress.json").read_text(encoding="utf-8"))
check("Retrieval recall@1", 0.94, rs["retrieval"]["recall_at_1"], tol=0.0001)
check("Retrieval recall@5", 1.00, rs["retrieval"]["recall_at_5"], tol=0.0001)
check("Retrieval MRR", 0.959, rs["retrieval"]["mrr"], tol=0.0001)
rc = pd.read_csv(P1 / "retrieval_stress.csv")
gold = rc[(rc.block == "model") & (rc.context == "gold") & (rc.seed.astype(str) == "mean")].iloc[0]
ret = rc[(rc.block == "model") & (rc.context == "retrieved") & (rc.seed.astype(str) == "mean")].iloc[0]
check("Retrieval gold AUROC", 0.5446, gold["auroc"])
check("Retrieval retrieved AUROC", 0.5212, ret["auroc"])
check("Retrieval gold F1", 0.3446, gold["f1"])
check("Retrieval retrieved F1", 0.3486, ret["f1"])

# ---------------------------------------------------------------------------
# P1 conformal (disjoint Platt/conformal split, mean over seeds)
# ---------------------------------------------------------------------------
conf = json.loads((P1 / "conformal.json").read_text(encoding="utf-8"))
dis = conf["sensitivity"]["disjoint_platt"]
for alpha, qa_cov, qa_pos, fb_cov, fb_pos in (
    ("0.1", 0.944, 0.685, 0.320, 0.003),
    ("0.05", 0.972, 0.842, 0.323, 0.006),
):
    qa_cov_vals = [dis[s][alpha]["ragtruth_qa_test"]["coverage"] for s in ("42", "123", "456")]
    qa_pos_vals = [dis[s][alpha]["ragtruth_qa_test"]["coverage_pos"] for s in ("42", "123", "456")]
    fb_cov_vals = [dis[s][alpha]["faithbench"]["coverage"] for s in ("42", "123", "456")]
    fb_pos_vals = [dis[s][alpha]["faithbench"]["coverage_pos"] for s in ("42", "123", "456")]
    check(f"Conformal {alpha} QA coverage", qa_cov, mean_json(qa_cov_vals), tol=0.002)
    check(f"Conformal {alpha} QA positive", qa_pos, mean_json(qa_pos_vals), tol=0.002)
    check(f"Conformal {alpha} FaithBench coverage", fb_cov, mean_json(fb_cov_vals), tol=0.002)
    check(f"Conformal {alpha} FaithBench positive", fb_pos, mean_json(fb_pos_vals), tol=0.002)
cc = conf["sensitivity"]["class_conditional"]
for alpha, pos in (("0.1", 0.887), ("0.05", 0.925)):
    vals = [cc[s][alpha]["ragtruth_qa_test"]["coverage_pos"] for s in ("42", "123", "456")]
    check(f"Conformal class-cond {alpha} QA positive", pos, mean_json(vals), tol=0.002)

# ---------------------------------------------------------------------------
# P1 multicalibration
# ---------------------------------------------------------------------------
mc = pd.read_csv(P1 / "multicalibration.csv")
qa_overall = mc[(mc.dataset == "ragtruth_qa_test") & (mc.binning == "overall") &
                (mc.score == "target_platt") & (mc.seed.astype(str) == "pooled")].iloc[0]
qa_short = mc[(mc.dataset == "ragtruth_qa_test") & (mc.binning == "length_words_natural") &
              (mc.bin == "<=60") & (mc.score == "target_platt") & (mc.seed.astype(str) == "pooled")].iloc[0]
qa_claim1 = mc[(mc.dataset == "ragtruth_qa_test") & (mc.binning == "claim_count") &
               (mc.bin == "1") & (mc.score == "target_platt") & (mc.seed.astype(str) == "pooled")].iloc[0]
fb_overall = mc[(mc.dataset == "faithbench") & (mc.binning == "overall") &
                (mc.score == "target_platt") & (mc.seed.astype(str) == "pooled")].iloc[0]
check("Multical QA overall ECE", 0.1251, qa_overall["ece"])
check("Multical QA short ECE", 0.2750, qa_short["ece"])
check("Multical QA one-claim ECE", 0.3463, qa_claim1["ece"])
check("Multical FaithBench target ECE", 0.6291, fb_overall["ece"])

# ---------------------------------------------------------------------------
out = pd.DataFrame(rows)
out.to_csv(OUT / "p1_number_audit.csv", index=False)
summary = {
    "n_checks": int(len(out)),
    "n_match": int((out.status == "match").sum()),
    "n_mismatch": int((out.status == "MISMATCH").sum()),
}
print(json.dumps(summary, indent=2))
if summary["n_mismatch"]:
    print(out[out.status == "MISMATCH"].to_string(index=False))
    raise SystemExit(1)
