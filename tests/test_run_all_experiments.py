"""B6 reproducibility-protocol tests: config validation, placeholders, dry-run,
abort/keep-going behavior, and the run-all report."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.run_all_experiments import (  # noqa: E402
    DEFAULT_CONFIG,
    git_commit,
    load_config,
    main,
    resolve_args,
)

CONFIG = ROOT / "configs" / "version_b.yaml"


def test_default_config_is_valid_and_complete():
    cfg = load_config(DEFAULT_CONFIG)
    ids = [p["id"] for p in cfg["phases"]]
    for required in ("prepare_halueval", "b1_unified", "extract_features", "b2", "b3", "b4", "b5", "manifest", "verify"):
        assert required in ids, f"missing phase {required}"
    assert ids == sorted(set(ids)) or len(ids) == len(set(ids))  # unique


def test_all_configured_scripts_exist_in_repo():
    cfg = load_config(DEFAULT_CONFIG)
    for p in cfg["phases"]:
        assert (ROOT / p["script"]).exists(), f"{p['script']} missing"


def test_placeholder_resolution():
    args = ["--device", "{device}", "--batch-size", "{batch_size}", "--resume"]
    resolved = resolve_args(args, {"device": "cpu", "batch_size": 256}, {"device": "cuda"})
    assert resolved == ["--device", "cuda", "--batch-size", "256", "--resume"]


def test_dry_run_prints_resolved_plan():
    out = subprocess.run(
        [sys.executable, str(ROOT / "src/models/run_all_experiments.py"),
         "--config", str(CONFIG), "--dry-run", "--device", "cpu"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "DRY RUN" in out.stdout
    assert "b3" in out.stdout and "run_b3_cross_domain.py --device cpu" in out.stdout


def test_main_from_to_filters_phases(tmp_path):
    report = main([
        "--config", str(CONFIG), "--from", "b3", "--to", "b4",
        "--dry-run",
    ])
    ids = [p["id"] for p in report["phases"]]
    assert ids == ["b3", "b4"]


def test_run_all_report_written_on_real_run(tmp_path, monkeypatch):
    """Simulate a tiny config run with stubbed subprocesses; verify report."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "schema: b6-run-all-v1\n"
        "defaults: {device: cpu, batch_size: 8}\n"
        "phases:\n"
        "  - {id: a, script: src/models/verify_artifacts.py, args: []}\n"
        "  - {id: b, script: src/models/verify_artifacts.py, args: []}\n",
        encoding="utf-8",
    )
    import src.models.run_all_experiments as m

    calls = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd, cwd):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    report = m.main(["--config", str(cfg), "--report", str(tmp_path / "report.json")])
    assert report["overall_status"] == "ok"
    assert [p["id"] for p in report["phases"]] == ["a", "b"]
    assert all(p["status"] == "ok" for p in report["phases"])
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["schema"] == "b6-run-all-report-v1"
    assert saved["resolved_defaults"] == {"device": "cpu", "batch_size": 8}
    assert len(calls) == 2


def test_failure_aborts_unless_keep_going(tmp_path, monkeypatch):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "schema: b6-run-all-v1\n"
        "defaults: {device: cpu, batch_size: 8}\n"
        "phases:\n"
        "  - {id: a, script: src/models/verify_artifacts.py, args: []}\n"
        "  - {id: b, script: src/models/verify_artifacts.py, args: []}\n"
        "  - {id: c, script: src/models/verify_artifacts.py, args: []}\n",
        encoding="utf-8",
    )
    import src.models.run_all_experiments as m

    calls = []

    class FakeResult:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd):
        calls.append(cmd)
        return FakeResult(1 if len(calls) == 1 else 0)

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    report = m.main(["--config", str(cfg), "--report", str(tmp_path / "abort.json")])
    assert report["overall_status"] == "failed"
    assert len(calls) == 1  # aborted after first failure

    calls.clear()
    report = m.main(["--config", str(cfg), "--keep-going", "--report", str(tmp_path / "keep.json")])
    assert report["overall_status"] == "failed"
    assert len(calls) == 3  # keep-going ran everything
    assert [p["status"] for p in report["phases"]] == ["failed", "ok", "ok"]


def test_invalid_config_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("schema: b6-run-all-v1\nphases: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--config", str(cfg), "--dry-run"])


def test_git_commit_returns_string_or_none():
    c = git_commit()
    assert c is None or (isinstance(c, str) and len(c) >= 7)
