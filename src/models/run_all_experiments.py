"""
B6 — Reproducible publication protocol: config-driven orchestrator (blueprint §11).

Runs the full Version B pipeline (B1 data -> B2 baselines -> B3 cross-domain ->
B4 calibration shift -> B5 explanation reliability -> manifest -> verify) by
subprocess-invoking the existing per-phase runners. The per-phase runners stay
the single source of truth; this script only orders them and records what ran.

Config: configs/version_b.yaml (schema b6-run-all-v1). Placeholders {device}
and {batch_size} are resolved from config defaults or CLI overrides.

Usage (repo root):
  python src/models/run_all_experiments.py --dry-run
  python src/models/run_all_experiments.py --device cuda --from b2 --to b5
  python src/models/run_all_experiments.py --keep-going

Every phase writes a status + duration into artifacts/results/run_all_report.json
(git commit included when run inside a git clone; null otherwise, matching the
per-phase run configs).
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_all_experiments")

CONFIG_SCHEMA = "b6-run-all-v1"
DEFAULT_CONFIG = ROOT / "configs" / "version_b.yaml"
DEFAULT_REPORT = ROOT / "artifacts" / "results" / "run_all_report.json"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=ROOT
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"{path} is not a {CONFIG_SCHEMA} config")
    phases = cfg.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("config must declare a non-empty phases list")
    ids = [p.get("id") for p in phases]
    if any(not i or ids.count(i) != 1 for i in ids):
        raise ValueError(f"phase ids must be unique and non-empty: {ids}")
    for p in phases:
        script = p.get("script")
        if not script or not (ROOT / script).exists():
            raise ValueError(f"phase '{p.get('id')}': script not found: {script}")
        if not isinstance(p.get("args", []), list):
            raise ValueError(f"phase '{p.get('id')}': args must be a list")
    return cfg


def resolve_args(args: list, defaults: dict, overrides: dict) -> list:
    tokens = {k: overrides.get(k, defaults.get(k)) for k in ("device", "batch_size")}
    return [str(a).format(**tokens) if isinstance(a, str) else a for a in args]


def run_phase(phase: dict, args: list, keep_going: bool) -> dict:
    script = ROOT / phase["script"]
    cmd = [sys.executable, str(script), *args]
    logger.info("==> %s: %s", phase["id"], " ".join(str(c) for c in cmd))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=ROOT)
        exit_code, status = result.returncode, "ok" if result.returncode == 0 else "failed"
    except OSError as e:
        exit_code, status = -1, f"spawn failed: {e}"
    return {
        "id": phase["id"],
        "script": phase["script"],
        "args": args,
        "exit_code": exit_code,
        "status": status,
        "duration_s": round(time.time() - t0, 2),
        "started_at_utc": started,
    }


def main(argv: list | None = None) -> dict:
    parser = argparse.ArgumentParser(description="B6 config-driven Version B pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to YAML config")
    parser.add_argument("--from", dest="from_phase", default=None, help="start phase id (inclusive)")
    parser.add_argument("--to", dest="to_phase", default=None, help="end phase id (inclusive)")
    parser.add_argument("--device", default=None, help="override device (cpu|cuda|auto)")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed phase")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="output report path")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    phases = cfg["phases"]
    ids = [p["id"] for p in phases]
    start = ids.index(args.from_phase) if args.from_phase else 0
    end = ids.index(args.to_phase) + 1 if args.to_phase else len(ids)
    if start > end:
        raise ValueError(f"--from {args.from_phase} comes after --to {args.to_phase}")
    phases = phases[start:end]

    overrides = {k: v for k, v in {"device": args.device, "batch_size": args.batch_size}.items() if v is not None}
    defaults = cfg.get("defaults", {})
    plan = [
        {"id": p["id"], "script": p["script"], "args": resolve_args(p.get("args", []), defaults, overrides),
         "description": p.get("description", "")}
        for p in phases
    ]

    if args.dry_run:
        print(f"DRY RUN ({len(plan)} phases, device={overrides.get('device', defaults.get('device'))}, "
              f"batch_size={overrides.get('batch_size', defaults.get('batch_size'))})")
        for p in plan:
            print(f"  [{p['id']:>20}] {p['script']} {' '.join(p['args'])}")
        return {"dry_run": True, "phases": plan}

    results = []
    overall = "ok"
    for phase in plan:
        result = run_phase(phase, phase["args"], args.keep_going)
        results.append(result)
        if result["status"] != "ok":
            overall = "failed"
            if not args.keep_going:
                logger.error("phase %s failed (exit %s); aborting (use --keep-going to continue)",
                             phase["id"], result["exit_code"])
                break

    report = {
        "schema": "b6-run-all-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": str(Path(args.config)),
        "resolved_defaults": {**defaults, **overrides},
        "overall_status": overall,
        "phases": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s (overall: %s)", report_path, overall)
    return report


if __name__ == "__main__":
    report = main()
    ok = report.get("dry_run") or report.get("overall_status") == "ok"
    sys.exit(0 if ok else 1)
