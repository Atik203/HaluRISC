"""
T2: claim-verification evaluation set + metrics (roadmap B7.5 T2).

Two modes:

  --build      sample N answers (from b5_review_cases.json), split into
               claims, run the NLI verifier, and write a human-labeling CSV:
               artifacts/results/b5/claim_eval_cases.csv
               (human_verdict column starts empty — fill by hand, B5.5-style)

  --evaluate   compare a filled sheet (human_verdict) against model_verdict:
               per-class agreement, binary precision/recall of "flagged"
               (contradicted|unsupported) claims vs human "problem" labels,
               and the contingency table. Descriptive only — no fixed
               pass thresholds (B5.7).

Run (repo root, .venv; --build needs the NLI models loaded via the API stack):
  python src/models/eval_claims.py --build --n 40
  python src/models/eval_claims.py --evaluate artifacts/results/b5/claim_eval_cases_reviewed.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

B5_REVIEW_CASES = ROOT / "artifacts" / "results" / "b5" / "b5_review_cases.json"
EVAL_CSV = ROOT / "artifacts" / "results" / "b5" / "claim_eval_cases.csv"

HUMAN_LABELS = {"supported", "contradicted", "unsupported", ""}  # "" = not yet reviewed

COLUMNS = ["sample_id", "claim_id", "claim_text", "context", "model_verdict",
           "model_confidence", "human_verdict", "notes"]


def _build_eval_set(n: int = 40, seed: int = 42) -> list:
    """Sample answers -> claims -> verdicts (requires NLI models + contexts)."""
    from src.api.main import load_feature_models  # reuse the API's model loading
    from src.claims.decompose import split_claims
    from src.claims.verify import verify_claims

    cases = json.loads(B5_REVIEW_CASES.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(cases), size=min(n, len(cases)), replace=False)

    nli = load_feature_models().get("nli")
    if nli is None:
        raise RuntimeError("NLI model not loaded (start via the API or HALU_API_PRELOAD=1)")

    rows = []
    for i in picks:
        case = cases[int(i)]
        claims = split_claims(case.get("answer", ""))
        if not claims:
            continue
        result = verify_claims(claims, case.get("context", ""), nli)
        for c in result["claims"]:
            rows.append({
                "sample_id": case["sample_id"],
                "claim_id": c["id"],
                "claim_text": c["text"],
                "context": (case.get("context") or "")[:2000],
                "model_verdict": c["verdict"],
                "model_confidence": c["confidence"],
                "human_verdict": "",
                "notes": "",
            })
    return rows


def write_eval_csv(rows: list, path: Path = EVAL_CSV) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} claim cases to {path}")
    print("Fill the human_verdict column (supported|contradicted|unsupported) by hand,")
    print("save as claim_eval_cases_reviewed.csv, then run --evaluate on it.")


def flagged(verdict: str) -> bool:
    return verdict in ("contradicted", "unsupported")


def evaluate_sheet(path: Path) -> dict:
    import csv

    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human_verdict") or "").strip()]

    reviewed = [r for r in rows if (r.get("human_verdict") or "").strip() in HUMAN_LABELS - {""}]
    if not reviewed:
        print(f"No reviewed rows in {path} (human_verdict empty).")
        return {}

    contingency = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    per_class_agreement = {}
    for v in ("supported", "contradicted", "unsupported"):
        sub = [r for r in reviewed if r["human_verdict"] == v]
        agree = sum(1 for r in sub if r["model_verdict"] == v)
        per_class_agreement[v] = {"n": len(sub), "model_agrees": agree}

    for r in reviewed:
        human_flag = flagged(r["human_verdict"])
        model_flag = flagged(r["model_verdict"])
        if model_flag and human_flag:
            contingency["tp"] += 1
        elif model_flag and not human_flag:
            contingency["fp"] += 1
        elif not model_flag and not human_flag:
            contingency["tn"] += 1
        else:
            contingency["fn"] += 1

    tp, fp, tn, fn = (contingency[k] for k in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else None
    overall_agree = sum(1 for r in reviewed if r["model_verdict"] == r["human_verdict"]) / len(reviewed)

    print(f"Reviewed claims: {len(reviewed)}")
    for v, stats in per_class_agreement.items():
        print(f"  {v:<14} n={stats['n']:<4} model agrees={stats['model_agrees']}")
    print(f"Binary flagging (contradicted|unsupported as 'problem'):")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  precision={precision:.3f}" if precision is not None else "  precision=n/a")
    print(f"  recall={recall:.3f}" if recall is not None else "  recall=n/a")
    print(f"  accuracy={accuracy:.3f}" if accuracy is not None else "  accuracy=n/a")
    print(f"  overall verdict agreement={overall_agree:.3f}")
    return {"contingency": contingency, "per_class_agreement": per_class_agreement,
            "precision": precision, "recall": recall, "accuracy": accuracy,
            "overall_agreement": overall_agree}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="T2 claim-verification evaluation set + metrics")
    parser.add_argument("--build", action="store_true", help="build the human-labeling CSV")
    parser.add_argument("--n", type=int, default=40, help="answers to sample for --build")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", metavar="CSV", default=None, help="evaluate a reviewed sheet")
    args = parser.parse_args(argv)

    if args.build:
        write_eval_csv(_build_eval_set(n=args.n, seed=args.seed))
        return 0
    if args.evaluate:
        evaluate_sheet(Path(args.evaluate))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
