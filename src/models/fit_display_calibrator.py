"""Fit the display calibrator for the API on natural-style RAGTruth QA data.

The B2 model saturates near 1.0 on full-sentence inputs because it was trained
on HaluEval's terse synthetic answers. The deployed API therefore uses a
separate calibrator fitted on natural RAGTruth QA responses (the B4 target
calibration set) so the shown percentage reflects the evidence style of real
chat answers.

Fits Platt, isotonic, and temperature scaling on 5,034 RAGTruth QA rows and
evaluates on the disjoint 900-row test set. Saves the best method (lowest
ECE, tie-break NLL) plus a metrics report. Run from the repo root:

  .venv/Scripts/python.exe src/models/fit_display_calibrator.py

Artifacts written:
  artifacts/models/b4/calibrator_display.joblib
  artifacts/models/b4/display_calibration.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
CAL_FILE = ROOT / "artifacts" / "results" / "b4" / "_stages" / "qa_cal_clean.parquet"
TEST_FILE = ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet"
OUT_MODEL = ROOT / "artifacts" / "models" / "b4" / "calibrator_display.joblib"
OUT_META = ROOT / "artifacts" / "models" / "b4" / "display_calibration.json"
BINS = 10


def ece(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        total += m.sum() / n * abs(acc - conf)
    return float(total)


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    cal = pd.read_parquet(CAL_FILE)
    test = pd.read_parquet(TEST_FILE)
    test = test[(test.subset == "ragtruth_qa_test") & (test.method == "raw")]

    y_cal = cal["label"].to_numpy()
    p_cal = cal["score_42"].to_numpy()
    y_test = test["label"].to_numpy()
    p_test = test["score"].to_numpy()

    z_cal, z_test = logit(p_cal), logit(p_test)

    platt = LogisticRegression(C=1e6)
    platt.fit(z_cal.reshape(-1, 1), y_cal)
    p_platt = platt.predict_proba(z_test.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p_cal, y_cal)
    p_iso = iso.predict(p_test)

    def temp_nll(t: float) -> float:
        return nll(y_cal, sigmoid(z_cal / t))

    res = minimize_scalar(temp_nll, bounds=(0.05, 5.0), method="bounded")
    t_best = res.x
    p_temp = sigmoid(z_test / t_best)

    methods = {
        "platt": (p_platt, platt),
        "isotonic": (p_iso, iso),
        "temperature": (p_temp, t_best),
    }

    report = {
        "n_calibration_rows": int(len(y_cal)),
        "n_test_rows": int(len(y_test)),
        "label_positive_rate_test": float(y_test.mean()),
        "methods": {},
    }
    for name, (p, _) in methods.items():
        report["methods"][name] = {
            "ece": ece(y_test, p),
            "brier": brier_score_loss(y_test, p),
            "nll": nll(y_test, p),
            "pred_positive_rate": float((p >= 0.5).mean()),
            "score_p50": float(np.median(p)),
            "score_p90": float(np.quantile(p, 0.9)),
            "score_mean": float(p.mean()),
        }
    report["raw_reference"] = {
        "ece": ece(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
        "nll": nll(y_test, p_test),
        "pred_positive_rate": float((p_test >= 0.5).mean()),
        "score_p50": float(np.median(p_test)),
    }

    best = min(report["methods"], key=lambda m: (report["methods"][m]["ece"], report["methods"][m]["nll"]))
    report["chosen"] = best

    import joblib

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"method": best, "calibrator": methods[best][1]}, OUT_MODEL)
    OUT_META.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {OUT_MODEL} (chosen: {best})")


if __name__ == "__main__":
    main()
