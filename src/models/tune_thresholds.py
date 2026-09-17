"""T4: feedback-driven NLI threshold tuning (report-only by default).

Searches the entailment/contradiction threshold grid for the combination that
maximizes verdict agreement on LABELED feedback rows (rows with a
`correct_verdict` field — filled manually or exported from the claim-eval
sheet). Rows with only agree/disagree feedback are reported as unresolved.

  python src/models/tune_thresholds.py                    # report only
  python src/models/tune_thresholds.py --apply            # write thresholds
  python src/models/tune_thresholds.py --from eval.csv    # use a reviewed sheet

--apply writes data/processed/verdict_thresholds.json which the verifier
reads at runtime (a deliberate, explicit human step; default thresholds stay
entail=0.5 / contra=0.5).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEEDBACK_LOG = ROOT / "data" / "processed" / "feedback_log.jsonl"
THRESHOLDS_FILE = ROOT / "data" / "processed" / "verdict_thresholds.json"
DEFAULT_THRESHOLDS = {"entail": 0.5, "contra": 0.5}

GRID = np.arange(0.35, 0.81, 0.05)


def load_rows(source: Path | None = None) -> list:
    """Feedback rows with claim_text + correct_verdict (+ evidence_sentence)."""
    rows = []
    if source is None:
        path = FEEDBACK_LOG
        if not path.exists():
            return []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    else:
        with open(source, encoding="utf-8", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    return rows


def labeled_rows(rows: list) -> list:
    out = []
    for r in rows:
        cv = (r.get("correct_verdict") or "").strip()
        if cv in ("supported", "contradicted", "unsupported") and (r.get("claim_text") or "").strip():
            out.append({**r, "correct_verdict": cv})
    return out


def _verdict(probs: np.ndarray, ent_thr: float, con_thr: float) -> str:
    contra, entail = probs[0], probs[1]
    if entail >= ent_thr and entail >= contra:
        return "supported"
    if contra >= con_thr:
        return "contradicted"
    return "unsupported"


def tune(rows: list, nli_model) -> dict:
    """Grid search over thresholds; returns {best, grid, unresolved}."""
    labeled = labeled_rows(rows)
    unresolved = [r for r in rows if r not in labeled]
    results = []
    if labeled:
        for ent in GRID:
            for con in GRID:
                agree = 0
                for r in labeled:
                    probs = np.asarray(nli_model.predict(
                        [[r.get("evidence_sentence", ""), r["claim_text"]]],
                        batch_size=1, apply_softmax=True))[0]
                    agree += int(_verdict(probs, float(ent), float(con)) == r["correct_verdict"])
                results.append({"entail": round(float(ent), 2), "contra": round(float(con), 2),
                                "agreement": agree / len(labeled)})
        best = max(results, key=lambda x: x["agreement"])
    else:
        best = None
    return {"best": best, "grid": results, "n_labeled": len(labeled),
            "n_unresolved": len(unresolved), "defaults": DEFAULT_THRESHOLDS}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="T4 feedback-driven NLI threshold tuning (report-only by default)")
    parser.add_argument("--apply", action="store_true", help="write the best thresholds (explicit human step)")
    parser.add_argument("--from", dest="source", default=None, help="CSV source instead of the feedback log")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.source) if args.source else None)
    if not rows:
        print(f"No feedback rows found ({args.source or FEEDBACK_LOG}). Collect feedback first, or pass --from.")
        return 1

    from src.features.extract_features import load_heavy_models

    print("Loading NLI model (first run downloads it)...")
    nli = load_heavy_models(device="cpu").get("nli")
    result = tune(rows, nli)

    print(f"Labeled rows: {result['n_labeled']}  | unresolved (agree/disagree only): {result['n_unresolved']}")
    if result["best"]:
        b = result["best"]
        print(f"Best thresholds: entail={b['entail']} contra={b['contra']} agreement={b['agreement']:.3f}")
        if args.apply:
            THRESHOLDS_FILE.write_text(json.dumps(
                {"entail": b["entail"], "contra": b["contra"],
                 "agreement": round(b["agreement"], 4),
                 "n_labeled": result["n_labeled"]}, indent=2), encoding="utf-8")
            print(f"Wrote {THRESHOLDS_FILE} (verifier reads it at runtime).")
    else:
        print("No labeled rows with correct_verdict — add labels (claim-eval sheet) before tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
