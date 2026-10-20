"""Fresh GPT judge comparison against EC-XGB on the same 200 rows (P2, G1-G3).

The saved 2026 comparison used the B2 calibrated model, not EC-XGB. This script
re-runs GPT 5.6 Luna on the same balanced 200-row sample with temperature 0,
pairs it with EC-XGB (m3, seed 42), and adds bootstrap CIs plus the length-only
baseline. Cost is about $0.02.

Run (repo root, .env with OPENAI_API_KEY):
  python src/models/review_llm_judge_gpt_ec_xgb.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

from src.models.config import FEATURES_FULL, RESULTS_DIR, SAMPLE_SEED
from src.models.eval_llm_judge import JUDGE_SYSTEM, load_test_set

ROOT = Path(__file__).resolve().parents[2]
B6 = RESULTS_DIR / "b6" / "b6_predictions.parquet"
OUT = RESULTS_DIR / "review"
N_SAMPLES = 200
N_BOOT = 1000
load_dotenv()


def gpt_call(client, model, q, c, a):
    user = f"Question: {q}\nContext: {c or '(none)'}\nAnswer: {a}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_completion_tokens=1000,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content.strip()
    try:
        chunk = content[content.find("{"): content.rfind("}") + 1]
        data = json.loads(chunk)
        judgment = str(data.get("judgment", "grounded")).lower()
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        judgment = "hallucinated" if "hallucinated" in content.lower() else "grounded"
        confidence = 0.5
    return (1 if judgment == "hallucinated" else 0), confidence, resp.usage


def bootstrap_ci(y, pred, metric, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], pred[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)

    test, _ = load_test_set()
    y = test["label"].to_numpy()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(SAMPLE_SEED)
    half = N_SAMPLES // 2
    sample_idx = np.concatenate([rng.choice(pos, half, replace=False), rng.choice(neg, half, replace=False)])
    sample = test.iloc[sample_idx].reset_index(drop=True)

    rows = []
    in_tokens = out_tokens = 0
    for i, row in sample.iterrows():
        t0 = time.perf_counter()
        pred, conf, usage = gpt_call(client, model, row["question"], row["context"], row["answer"])
        latency = (time.perf_counter() - t0) * 1000
        in_tokens += usage.prompt_tokens
        out_tokens += usage.completion_tokens
        rows.append({"sample_id": row["sample_id"], "label": int(row["label"]),
                     "gpt_pred": pred, "gpt_conf": conf, "latency_ms": latency})
        if (i + 1) % 25 == 0:
            print(f"judged {i + 1}/{len(sample)}")

    gdf = pd.DataFrame(rows)
    y_g = gdf["gpt_pred"].to_numpy()
    y_sub = sample["label"].to_numpy()

    b6 = pd.read_parquet(B6)
    b6 = b6[(b6["variant"] == "m3") & (b6["seed"] == 42) & (b6["dataset"] == "halueval_test")]
    b6 = b6.set_index("sample_id")
    ec_score = b6.loc[sample["sample_id"], "score"].to_numpy()
    ec_pred = (ec_score >= 0.5).astype(int)

    feat = pd.read_parquet(FEATURES_FULL, columns=["sample_id", "label", "split", "n_chars", "n_words"])
    train = feat[feat["split"] == "train"]
    cols = ["n_chars", "n_words"]
    Xtr = np.column_stack([np.log1p(train[c].to_numpy(dtype=float)) for c in cols])
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), train["label"].to_numpy())
    test_feat = feat.set_index("sample_id").loc[sample["sample_id"]]
    Xte = np.column_stack([np.log1p(test_feat[c].to_numpy(dtype=float)) for c in cols])
    len_score = lr.predict_proba(scaler.transform(Xte))[:, 1]
    len_pred = (len_score >= 0.5).astype(int)

    def block(pred):
        return {
            "accuracy": float(accuracy_score(y_sub, pred)),
            "precision": float(precision_score(y_sub, pred, zero_division=0)),
            "recall": float(recall_score(y_sub, pred, zero_division=0)),
            "f1": float(f1_score(y_sub, pred, zero_division=0)),
        }

    gpt_metrics = block(y_g)
    gpt_metrics["f1_ci95"] = bootstrap_ci(y_sub, y_g, lambda a, b: f1_score(a, b, zero_division=0))
    gpt_metrics["precision_ci95"] = bootstrap_ci(y_sub, y_g, lambda a, b: precision_score(a, b, zero_division=0))
    gpt_metrics["recall_ci95"] = bootstrap_ci(y_sub, y_g, lambda a, b: recall_score(a, b, zero_division=0))

    judge_wrong = y_g != y_sub
    ec_wrong = ec_pred != y_sub
    table = [[int((judge_wrong & ec_wrong).sum()), int((judge_wrong & ~ec_wrong).sum())],
             [int((~judge_wrong & ec_wrong).sum()), int((~judge_wrong & ~ec_wrong).sum())]]
    p = float(mcnemar(table, exact=False, correction=True).pvalue)
    cost = (in_tokens / 1e6) * 0.20 + (out_tokens / 1e6) * 1.20

    bundle = {
        "meta": {
            "source": "review_llm_judge_gpt_ec_xgb.py",
            "model": model,
            "temperature": 0,
            "n_samples": N_SAMPLES,
            "sample_prevalence": float(y_sub.mean()),
            "sample_seed": int(SAMPLE_SEED),
            "ec_xgb": "b6 m3 seed 42",
            "note": "Fresh run with EC-XGB; supersedes the B2-model judge table in llm_judge_results.json.",
        },
        "gpt": gpt_metrics,
        "ec_xgb_same_rows": {**block(ec_pred), "auroc": float(roc_auc_score(y_sub, ec_score))},
        "length_only_same_rows": {**block(len_pred), "auroc": float(roc_auc_score(y_sub, len_score))},
        "agreement_gpt_ec_xgb": float((y_g == ec_pred).mean()),
        "mcnemar_p": p,
        "discordant_pairs": {"gpt_wrong_ec_right": table[0][1], "gpt_right_ec_wrong": table[1][0]},
        "cost_usd": round(cost, 4),
        "cost_per_1000_usd": round(cost / len(sample) * 1000, 3),
        "tokens": {"input": int(in_tokens), "output": int(out_tokens)},
        "latency_ms_p50": float(np.median(gdf["latency_ms"])),
        "latency_ms_p95": float(np.percentile(gdf["latency_ms"], 95)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(OUT / "llm_judge_gpt_ec_xgb_per_sample.csv", index=False)
    (OUT / "llm_judge_gpt_ec_xgb.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({k: bundle[k] for k in ("gpt", "ec_xgb_same_rows", "length_only_same_rows",
                                             "agreement_gpt_ec_xgb", "mcnemar_p", "cost_usd")}, indent=2))


if __name__ == "__main__":
    main()
