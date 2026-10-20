"""Score the neural baseline and the length-only model on the same 200 judge rows.

Reads the saved neural-baseline predictions and the judge sample list, then
reports the same metrics as Table 12 so every system is compared on identical
rows. Writes artifacts/results/review/judge_rows_comparison.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.config import DATA_PROCESSED, RESULTS_DIR

RES = RESULTS_DIR
OUT = RES / "review"


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
    }


def main() -> None:
    judge = pd.read_csv(OUT / "llm_judge_gpt_ec_xgb_per_sample.csv")
    sample_ids = judge["sample_id"].tolist()
    labels = judge["label"].to_numpy()

    neural = pd.read_parquet(OUT / "neural_baseline" / "neural_baseline_predictions.parquet")
    neural = neural[(neural["dataset"] == "halueval_test") & (neural["seed"] == 42)].set_index("sample_id")
    neural_p = neural.loc[sample_ids, "score_raw"].to_numpy()
    neural_cal = neural.loc[sample_ids, "score_calibrated"].to_numpy()

    b6 = pd.read_parquet(RES / "b6" / "b6_predictions.parquet")
    b6 = b6[(b6["variant"] == "m3") & (b6["seed"] == 42) & (b6["dataset"] == "halueval_test")].set_index("sample_id")
    ec_p = b6.loc[sample_ids, "score"].to_numpy()

    feat = pd.read_parquet(DATA_PROCESSED / "features_full.parquet", columns=["sample_id", "label", "split", "n_chars", "n_words"])
    train = feat[feat["split"] == "train"]
    cols = ["n_chars", "n_words"]
    Xtr = np.column_stack([np.log1p(train[c].to_numpy(dtype=float)) for c in cols])
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), train["label"].to_numpy())
    test_feat = feat.set_index("sample_id").loc[sample_ids]
    Xte = np.column_stack([np.log1p(test_feat[c].to_numpy(dtype=float)) for c in cols])
    len_p = lr.predict_proba(scaler.transform(Xte))[:, 1]

    qwen = pd.read_csv(OUT / "llm_judge_qwen_per_sample.csv").set_index("sample_id")
    qwen_p = qwen.loc[sample_ids, "qwen_pred"].to_numpy()
    gpt_p = judge["gpt_pred"].to_numpy()

    bundle = {
        "meta": {
            "source": "review_judge_rows_comparison.py",
            "n_rows": len(sample_ids),
            "prevalence": float(labels.mean()),
            "rows": "the same 200 balanced HaluEval rows used by both judges",
        },
        "ec_xgb": metrics(labels, ec_p),
        "roberta_base_raw": metrics(labels, neural_p),
        "roberta_base_calibrated": metrics(labels, neural_cal),
        "length_only": metrics(labels, len_p),
        "gpt_judge": metrics(labels, gpt_p),
        "qwen_judge": metrics(labels, qwen_p),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "judge_rows_comparison.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
