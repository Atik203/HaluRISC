"""Local Qwen 3.5 4B second judge on the same 200-sample judge set (P2, G4).

Uses Ollama's native /api/chat with thinking disabled, so the 4B model answers
the same JSON prompt as the GPT judge. Also evaluates EC-XGB (m3, seed 42) and
a length-only baseline on exactly the same rows, and adds bootstrap CIs.

Run (repo root, Ollama running):
  python src/models/review_llm_judge_qwen.py
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.config import FEATURES_FULL, RESULTS_DIR, SAMPLE_SEED
from src.models.eval_llm_judge import JUDGE_SYSTEM, load_test_set

ROOT = Path(__file__).resolve().parents[2]
B6 = RESULTS_DIR / "b6" / "b6_predictions.parquet"
OUT = RESULTS_DIR / "review"
MODEL = "qwen3.5:4b"
N_SAMPLES = 200
N_BOOT = 1000


def qwen_call(q: str, c: str, a: str) -> tuple[int, float, str, float]:
    user = f"Question: {q}\nContext: {c or '(none)'}\nAnswer: {a}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 200},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    latency = (time.perf_counter() - t0) * 1000
    content = (data.get("message", {}).get("content") or "").strip()
    judgment, confidence = "grounded", 0.5
    try:
        chunk = content[content.find("{"): content.rfind("}") + 1]
        parsed = json.loads(chunk)
        judgment = str(parsed.get("judgment", "grounded")).lower()
        confidence = float(parsed.get("confidence", 0.5))
    except Exception:
        if "hallucinated" in content.lower() and "grounded" not in content.lower():
            judgment = "hallucinated"
    return (1 if judgment == "hallucinated" else 0), confidence, content, latency


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
    test, _ = load_test_set()
    y = test["label"].to_numpy()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(SAMPLE_SEED)
    half = N_SAMPLES // 2
    sample_idx = np.concatenate([
        rng.choice(pos, half, replace=False),
        rng.choice(neg, half, replace=False),
    ])
    sample = test.iloc[sample_idx].reset_index(drop=True)
    y_sub = sample["label"].to_numpy()

    rows = []
    n_total = len(sample)
    t_start = time.time()
    pbar = tqdm(total=n_total, desc="Qwen judge", unit="sample", dynamic_ncols=True)
    for i, row in sample.iterrows():
        pred, conf, raw, latency = qwen_call(row["question"], row["context"], row["answer"])
        rows.append({
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "qwen_pred": pred,
            "qwen_conf": conf,
            "latency_ms": latency,
            "raw": raw,
        })
        done = i + 1
        pbar.update(1)
        if done % 25 == 0 or done == n_total:
            elapsed = time.time() - t_start
            eta = elapsed / done * (n_total - done)
            print(f"[{done}/{n_total}] {100 * done / n_total:.1f}% | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)
    pbar.close()

    qdf = pd.DataFrame(rows)
    y_q = qdf["qwen_pred"].to_numpy()

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

    qwen_metrics = block(y_q)
    qwen_metrics["f1_ci95"] = bootstrap_ci(y_sub, y_q, lambda a, b: f1_score(a, b, zero_division=0))
    qwen_metrics["precision_ci95"] = bootstrap_ci(y_sub, y_q, lambda a, b: precision_score(a, b, zero_division=0))
    qwen_metrics["recall_ci95"] = bootstrap_ci(y_sub, y_q, lambda a, b: recall_score(a, b, zero_division=0))

    bundle = {
        "meta": {
            "source": "review_llm_judge_qwen.py",
            "model": MODEL,
            "serving": "Ollama native /api/chat, think=false, temperature=0",
            "n_samples": N_SAMPLES,
            "sample_prevalence": float(y_sub.mean()),
            "sample_seed": int(SAMPLE_SEED),
            "ec_xgb": "b6 m3 seed 42",
            "gpt_reference": json.loads((RESULTS_DIR / "llm_judge_results.json").read_text()),
            "note": "GPT reference was produced by eval_llm_judge.py with the B2 calibrated model; this run uses EC-XGB and should be paired with a fresh GPT run before the paper table is final.",
        },
        "qwen": qwen_metrics,
        "ec_xgb_same_rows": {**block(ec_pred), "auroc": float(roc_auc_score(y_sub, ec_score))},
        "length_only_same_rows": {**block(len_pred), "auroc": float(roc_auc_score(y_sub, len_score))},
        "agreement_qwen_ec_xgb": float((y_q == ec_pred).mean()),
        "agreement_qwen_labels": float((y_q == y_sub).mean()),
        "latency_ms_p50": float(np.median(qdf["latency_ms"])),
        "latency_ms_p95": float(np.percentile(qdf["latency_ms"], 95)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(OUT / "llm_judge_qwen_per_sample.csv", index=False)
    (OUT / "llm_judge_qwen.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({k: bundle[k] for k in ("qwen", "ec_xgb_same_rows", "length_only_same_rows",
                                             "agreement_qwen_ec_xgb", "latency_ms_p50")}, indent=2))


if __name__ == "__main__":
    main()
