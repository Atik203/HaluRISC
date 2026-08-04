"""
HaluRISC LLM-as-judge comparison (blueprint A10 external baseline + cost table).

Runs GPT 5.6 Luna as a hallucination judge on a balanced sample of the test set
and compares against the XGBoost model: accuracy/precision/recall/F1, agreement,
latency, and a real token-cost estimate.

Cost: ~200 samples x ~1.1K tokens ≈ $0.05-0.15 depending on model pricing.
Override sample size with HALU_JUDGE_N.

Run (repo root, .venv):
  python src/models/eval_llm_judge.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_llm_judge")

load_dotenv()  # OPENAI_API_KEY, OPENAI_MODEL

from src.models.config import FEATURES_FALLBACK, FEATURES_FULL, MODELS_DIR, QA_CLEAN, RESULTS_DIR, SAMPLE_SEED

N_SAMPLES = int(os.environ.get("HALU_JUDGE_N", "200"))

PRICING = {"input_per_mtok": 0.20, "output_per_mtok": 1.20}  # GPT 5.6 Luna (roadmap Phase 5)


def load_test_set():
    path = FEATURES_FULL if FEATURES_FULL.exists() else FEATURES_FALLBACK
    df = pd.read_parquet(path)
    clean = pd.read_parquet(QA_CLEAN)
    text_cols = [c for c in ["question", "context", "answer"] if c in clean.columns]
    if text_cols:
        df = pd.concat([df, clean[text_cols]], axis=1)
    feature_cols = json.loads((MODELS_DIR / "feature_names.json").read_text())
    test = df[df["split"] == "test"].reset_index(drop=True)
    return test, feature_cols


def xgb_probabilities(test, feature_cols):
    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))[:, 1]

    else:
        predict_proba = bundle.predict_proba
    X = test[feature_cols].values
    return predict_proba(X)


JUDGE_SYSTEM = (
    "You are an expert hallucination-judge. Given a question, a reference context, and an answer, "
    "decide whether the answer contains hallucinated content (unsupported, contradictory, or fabricated "
    "information relative to the context). Respond with JSON only: "
    '{"judgment": "hallucinated"|"grounded", "confidence": 0.0-1.0}'
)


def judge_one(client, model_name, q, c, a):
    import json as _json
    import re

    user = f"Question: {q}\nContext: {c or '(none)'}\nAnswer: {a}"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=1000,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        data = _json.loads(content[content.find("{") : content.rfind("}") + 1])
    except _json.JSONDecodeError:
        judgment = "hallucinated" if "hallucinated" in content.lower() else "grounded"
        return judgment, 0.5, resp.usage
    judgment = data.get("judgment", "grounded")
    confidence = float(data.get("confidence", 0.5))
    return judgment, confidence, resp.usage


def main():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in .env — cannot run LLM judge.")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)

    test, feature_cols = load_test_set()
    y_prob_xgb = xgb_probabilities(test, feature_cols)
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    y_true = test["label"].values

    # Balanced sample (half hallucinated, half grounded)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    rng = np.random.default_rng(SAMPLE_SEED)
    n_half = N_SAMPLES // 2
    sample_idx = np.concatenate([rng.choice(pos, n_half, replace=False), rng.choice(neg, n_half, replace=False)])

    judgments, confs, latencies = [], [], []
    in_tokens = out_tokens = 0
    for i, idx in enumerate(sample_idx):
        row = test.iloc[idx]
        t0 = time.perf_counter()
        judgment, conf, usage = judge_one(client, model_name, row["question"], row["context"], row["answer"])
        latencies.append((time.perf_counter() - t0) * 1000)
        judgments.append(1 if judgment == "hallucinated" else 0)
        confs.append(conf)
        in_tokens += usage.prompt_tokens
        out_tokens += usage.completion_tokens
        if (i + 1) % 50 == 0:
            logger.info(f"judged {i + 1}/{len(sample_idx)}")

    y_pred_judge = np.array(judgments)
    y_true_sub = y_true[sample_idx]
    y_xgb_sub = y_pred_xgb[sample_idx]

    # McNemar: judge vs XGBoost on the same 200 samples (off-diagonal = discordant)
    judge_wrong = y_pred_judge != y_true_sub
    xgb_wrong = y_xgb_sub != y_true_sub
    both_wrong = int((judge_wrong & xgb_wrong).sum())
    judge_wrong_xgb_right = int((judge_wrong & ~xgb_wrong).sum())
    judge_right_xgb_wrong = int((~judge_wrong & xgb_wrong).sum())
    both_right = int((~judge_wrong & ~xgb_wrong).sum())
    from statsmodels.stats.contingency_tables import mcnemar

    mcn = mcnemar(
        [[both_wrong, judge_wrong_xgb_right], [judge_right_xgb_wrong, both_right]],
        exact=False,
        correction=True,
    )
    mcnemar_p = float(mcn.pvalue)

    cost = (in_tokens / 1e6) * PRICING["input_per_mtok"] + (out_tokens / 1e6) * PRICING["output_per_mtok"]

    results = {
        "n_samples": int(len(sample_idx)),
        "model": model_name,
        "judge": {
            "accuracy": round(float(accuracy_score(y_true_sub, y_pred_judge)), 4),
            "precision": round(float(precision_score(y_true_sub, y_pred_judge, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_sub, y_pred_judge, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_sub, y_pred_judge, zero_division=0)), 4),
            "latency_ms_p50": round(float(np.median(latencies)), 1),
            "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
        },
        "xgboost_on_same_subset": {
            "accuracy": round(float(accuracy_score(y_true_sub, y_xgb_sub)), 4),
            "precision": round(float(precision_score(y_true_sub, y_xgb_sub, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_sub, y_xgb_sub, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_sub, y_xgb_sub, zero_division=0)), 4),
        },
        "agreement_with_xgboost": round(float((y_pred_judge == y_xgb_sub).mean()), 4),
        "mcnemar_judge_vs_xgboost_p": mcnemar_p,
        "discordant_pairs": {"judge_wrong_xgb_right": judge_wrong_xgb_right, "judge_right_xgb_wrong": judge_right_xgb_wrong},
        "cost_usd": round(cost, 4),
        "cost_per_1000_usd": round(cost / len(sample_idx) * 1000, 3),
        "tokens": {"input": int(in_tokens), "output": int(out_tokens)},
        "pricing": PRICING,
    }

    with open(RESULTS_DIR / "llm_judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" LLM-as-Judge (GPT 5.6 Luna) vs XGBoost on {len(sample_idx)} test samples")
    print("=" * 80)
    for name, m in [("Judge", results["judge"]), ("XGBoost", results["xgboost_on_same_subset"])]:
        print(f"{name:<10} acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"Agreement: {results['agreement_with_xgboost']:.4f} | McNemar p={mcnemar_p:.2e} | "
          f"Judge cost: ${results['cost_usd']:.4f} ({results['cost_per_1000_usd']:.3f}/1K) | p50 {results['judge']['latency_ms_p50']}ms")
    print("=" * 80)
    logger.info(f"Saved llm_judge_results.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
