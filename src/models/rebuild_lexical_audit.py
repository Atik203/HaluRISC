"""WP7 supporting audit: length and overlap statistics on the HaluEval test split.

Writes artifacts/results/rebuild/halueval_lexical_audit.csv.

The audit explains why external lexical baselines are not transportable: the
answer-coverage overlap and the Jaccard overlap point in opposite directions,
and answer length alone separates the classes.

Run (repo root):
  python src/models/rebuild_lexical_audit.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features_full.parquet"
OUT = ROOT / "artifacts" / "results" / "rebuild" / "halueval_lexical_audit.csv"


def main() -> None:
    df = pd.read_parquet(FEATURES)
    test = df[df["split"] == "test"].copy()

    rows = []
    for label, name in ((0, "faithful"), (1, "hallucinated")):
        g = test[test["label"] == label]
        rows.extend(
            [
                {"statistic": "n_rows", "label": name, "value": len(g)},
                {"statistic": "mean_words", "label": name, "value": g["n_words"].mean()},
                {"statistic": "median_words", "label": name, "value": g["n_words"].median()},
                {"statistic": "mean_chars", "label": name, "value": g["n_chars"].mean()},
                {
                    "statistic": "mean_overlap_answer_context",
                    "label": name,
                    "value": g["overlap_answer_context"].mean(),
                },
                {
                    "statistic": "mean_jaccard_ans_ctx",
                    "label": name,
                    "value": g["jaccard_ans_ctx"].mean(),
                },
            ]
        )

    for feature in ("overlap_answer_context", "jaccard_ans_ctx", "n_words"):
        rows.append(
            {
                "statistic": "auroc_risk",
                "label": feature,
                "value": roc_auc_score(test["label"], test[feature]),
            }
        )
    rows.append(
        {
            "statistic": "auroc_risk",
            "label": "overlap_answer_context_flipped",
            "value": roc_auc_score(test["label"], -test["overlap_answer_context"]),
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.4f")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
