"""Source-indicator ablation for the review response.

Trains the deployed EC-XGB recipe (m3: 26 base + 8 claim aggregates + RAGTruth
non-QA training rows) with the source indicator removed from the feature
vector, holding the tuned m3 hyperparameters fixed per seed. Evaluates the
HaluEval test split and the external corpora exactly like run_b6_modified.py.

Writes artifacts/results/extras/source_ablation.json (new file only; the
frozen b6 artifacts are untouched).

Run from the repo root:

  .venv/Scripts/python.exe src/models/review_source_ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.run_b6_modified import (  # noqa: E402
    BASE_COLUMNS,
    CLAIM_COLUMNS,
    DEFAULT_ALPHA,
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
)

OUT = ROOT / "artifacts" / "results" / "extras"
SEEDS = (42, 123, 456)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(BASE_COLUMNS) + list(CLAIM_COLUMNS)  # 34: m3 minus the indicator
    constraints = constraint_string(names)
    tuning = json.loads(
        (ROOT / "artifacts" / "results" / "b6" / "b6_tuning.json").read_text(encoding="utf-8")
    )["m3"]

    halueval = load_halueval()
    external = load_external()
    ext = external_roles(external)
    extra = m3_train_rows(external)

    train_df = halueval[halueval["split"] == "train"].copy()
    train_df[SOURCE_COLUMN] = 0.0
    extra = extra.copy()
    extra[SOURCE_COLUMN] = 1.0
    import pandas as pd

    train_df = pd.concat([train_df, extra], ignore_index=True)
    val_df = halueval[halueval["split"] == "val"].reset_index(drop=True)
    test_df = halueval[halueval["split"] == "test"].reset_index(drop=True)

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    X_train = feature_matrix(train_df, names, True)
    X_val = feature_matrix(val_df, names, True)
    X_test = feature_matrix(test_df, names, True)

    print(f"ablated feature count: {len(names)} (m3 has {len(names) + 1})")
    print(f"train rows: {len(X_train)} (HaluEval {len(train_df) - len(extra)} + external {len(extra)})")

    rows = {}
    for seed in SEEDS:
        params = tuning[str(seed)]["params"]
        model = make_variant_xgb(params, seed, constraints, early_stopping=True)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_test = model.predict_proba(X_test)[:, 1]
        entry = {
            "params": params,
            "in_domain": in_domain_metrics(y_test, p_test, 0.5),
            "external": {},
        }
        for name, frame in ext.items():
            if len(frame) == 0:
                continue
            X = feature_matrix(frame, names, True)
            y = frame["label"].to_numpy()
            p = model.predict_proba(X)[:, 1]
            entry["external"][name] = external_metrics(y, p, 0.5)
        rows[str(seed)] = entry
        d = entry["in_domain"]
        print(
            f"seed {seed}: F1 {d['f1']:.4f} AUROC {d['auroc']:.4f} "
            f"| QA F1 {entry['external']['ragtruth_qa_test']['f1']:.4f} "
            f"flag {entry['external']['ragtruth_qa_test']['predicted_positive_rate']:.4f} "
            f"| all flag {entry['external']['ragtruth_all_test']['predicted_positive_rate']:.4f}"
        )

    def mean_std(key: str, sub: str | None = None):
        if sub is None:
            vals = [rows[s]["in_domain"][key] for s in rows]
        else:
            vals = [rows[s]["external"][sub][key] for s in rows if sub in rows[s]["external"]]
        vals = [v for v in vals if v is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "in_domain": {k: mean_std(k) for k in ("precision", "recall", "f1", "auroc", "mcc", "ece", "brier")},
        "external": {
            sub: {
                k: mean_std(k, sub)
                for k in ("recall", "precision", "f1", "auroc", "predicted_positive_rate", "ece", "brier")
            }
            for sub in ext
        },
    }
    payload = {
        "variant": "m3 minus source indicator",
        "n_features": len(names),
        "seeds": list(SEEDS),
        "hyperparameters": "held fixed at the tuned m3 values per seed",
        "per_seed": rows,
        "summary": summary,
    }
    path = OUT / "source_ablation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print("summary in-domain:", json.dumps(summary["in_domain"], indent=1))
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
