"""Standalone monotone-constraint ablation for the 2026-10-20 review (P2, F2).

The published m1 row bundles log-scaled lengths with the monotone constraints,
so it cannot isolate the constraint effect. This script trains the deployed
35-feature EC-XGB recipe with and without the constraints, same seeds, same
tuning parameters, same early stopping. Runs on the current 6 GB GPU.

Run (repo root):
  python src/models/review_monotone_ablation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.models.config import RESULTS_DIR
from src.models.run_b6_modified import (
    B6Config,
    SOURCE_COLUMN,
    constraint_string,
    external_metrics,
    external_roles,
    feature_matrix,
    in_domain_metrics,
    load_external,
    load_halueval,
    m3_train_rows,
    make_variant_xgb,
    variant_features,
)

SEEDS = [42, 123, 456]
OUT = RESULTS_DIR / "review"
TUNING = RESULTS_DIR / "b6" / "b6_tuning.json"


def main() -> None:
    cfg = B6Config(variants=("m3",), seeds=SEEDS)
    tuning = json.loads(TUNING.read_text(encoding="utf-8"))["m3"]

    halueval = load_halueval()
    external = load_external()
    ext = external_roles(external)
    ext_train = m3_train_rows(external)

    names = variant_features("m3")
    monotone = constraint_string(names)
    train_df = halueval[halueval["split"] == "train"].copy()
    train_df[SOURCE_COLUMN] = 0.0
    extra = ext_train.copy()
    extra[SOURCE_COLUMN] = 1.0
    train_df = pd.concat([train_df, extra], ignore_index=True)

    val_df = halueval[halueval["split"] == "val"].reset_index(drop=True)
    test_df = halueval[halueval["split"] == "test"].reset_index(drop=True)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    X_train = feature_matrix(train_df, names, log_lengths=True)
    X_val = feature_matrix(val_df, names, log_lengths=True)
    X_test = feature_matrix(test_df, names, log_lengths=True)

    rows = []
    for seed in SEEDS:
        params = tuning[str(seed)]["params"]
        for mode, constraint in (("monotone", monotone), ("none", None)):
            model = make_variant_xgb(params, seed, constraint, early_stopping=True)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            row = {
                "seed": seed,
                "mode": mode,
                "n_features": len(names),
                **{k: v for k, v in in_domain_metrics(y_test, model.predict_proba(X_test)[:, 1], 0.5).items()
                   if k != "confusion"},
            }
            for dataset, frame in ext.items():
                if len(frame) == 0:
                    continue
                p = model.predict_proba(feature_matrix(frame, names, log_lengths=True))[:, 1]
                m = external_metrics(frame["label"].to_numpy(), p, 0.5)
                row[f"{dataset}_f1"] = m["f1"]
                row[f"{dataset}_auroc"] = m["auroc"]
                row[f"{dataset}_flag"] = m["predicted_positive_rate"]
                row[f"{dataset}_ece"] = m["ece"]
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = df.groupby("mode")[
        [c for c in df.columns if c not in ("seed", "mode")]
    ].agg(["mean", "std"]).round(6)

    pivot = df.groupby("mode")[
        ["f1", "auroc", "ece", "ragtruth_all_test_auroc", "ragtruth_all_test_flag"]
    ].mean()
    delta = (pivot.loc["monotone"] - pivot.loc["none"]).round(6)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "monotone_ablation_per_seed.csv", index=False)
    summary.to_csv(OUT / "monotone_ablation_summary.csv")
    bundle = {
        "meta": {
            "source": "review_monotone_ablation.py",
            "variant": "m3 (35 features)",
            "seeds": SEEDS,
            "training_rows": int(len(train_df)),
            "tuning_params": {str(s): tuning[str(s)]["params"] for s in SEEDS},
            "constraints": monotone,
            "threshold": 0.5,
        },
        "per_seed": df.to_dict(orient="records"),
        "mean": pivot.round(6).to_dict(orient="index"),
        "delta_monotone_minus_none": delta.to_dict(),
    }
    (OUT / "monotone_ablation.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(pivot.round(4).to_string())
    print("\nDelta (monotone - none):")
    print(delta.to_string())
    print("\nSaved:", OUT / "monotone_ablation.json")


if __name__ == "__main__":
    main()
