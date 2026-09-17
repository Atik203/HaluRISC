"""B6 manifest schema tests: raw hashes, seeds, feature groups, b5 entries,
source fingerprint, exclusions, and no-secret guarantees."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.make_manifest import EXCLUDE_FRAGMENTS, raw_hashes  # noqa: E402


def _load_manifest() -> dict:
    return json.loads((ROOT / "artifacts" / "results" / "manifest.json").read_text())


def test_manifest_schema_complete():
    m = _load_manifest()
    for key in ("generated_at", "git_commit", "source_fingerprint", "model_version",
                "seeds", "feature_groups", "split_report", "dataset_sha256",
                "raw_sha256", "b_artifacts_sha256", "versions", "hardware", "artifacts"):
        assert key in m, f"manifest missing {key}"
    assert m["seeds"] == [42, 123, 456]
    assert len(m["feature_groups"]) == 7
    assert m["feature_groups"]["nli"][0].startswith("nli_")


def test_manifest_covers_b5_artifacts():
    m = _load_manifest()
    for name in ("b5_run_config.json", "b5_feature_importance.json",
                 "b5_neutralization.json", "b5_stability_bootstrap.json",
                 "b5_perturbation_aggregates.csv", "b5_review_cases.csv",
                 "b5_failure_cases.json"):
        assert m["b_artifacts_sha256"].get(name), f"manifest missing hash for {name}"


def test_manifest_raw_hashes_present():
    m = _load_manifest()
    raw = m["raw_sha256"]
    assert raw, "raw_sha256 must not be empty"
    joined = json.dumps(raw)
    assert "ragtruth_official" in joined or "qa_data.json" in joined
    for v in raw.values():
        assert v and len(v) == 64


def test_manifest_excludes_internal_paths():
    m = _load_manifest()
    for frag in ("_stages", "b2_smoke_test", "b2_test_tmp", "b5_crash.log", "__pycache__"):
        assert not any(frag in p for p in m["artifacts"]), f"{frag} leaked into artifacts list"


def test_manifest_has_no_secrets():
    """The manifest must never contain API keys or token patterns."""
    text = json.dumps(_load_manifest())
    for pattern in ("sk-", "OPENAI_API_KEY", "api_key", "DEEPSEEK_API_KEY"):
        assert pattern.lower() not in text.lower(), f"secret pattern {pattern} found in manifest"


def test_raw_hashes_walk():
    raw = raw_hashes()
    assert isinstance(raw, dict)
    assert any(p.endswith("revision.json") for p in raw)  # download revisions recorded


def test_source_fingerprint_env_passthrough(tmp_path, monkeypatch):
    """HALU_SOURCE_FINGERPRINT must land in the manifest when set."""
    import src.models.make_manifest as mm

    monkeypatch.setenv("HALU_SOURCE_FINGERPRINT", "abc123")
    monkeypatch.setattr(mm, "RESULTS_DIR", tmp_path)
    manifest = mm.main()
    assert manifest["source_fingerprint"] == "abc123"
    assert (tmp_path / "manifest.json").exists()


def test_no_secret_patterns_in_repo_config():
    """Regression: no API-key-like patterns anywhere in tracked source/configs."""
    import re

    bad = re.compile(r"sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY\s*=\s*\S+")
    hits = []
    for root in ("src", "configs", "colab"):
        for p in sorted((ROOT / root).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".txt", ".md"):
                if any(x in str(p) for x in ("node_modules", "__pycache__")):
                    continue
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if bad.search(line):
                        hits.append(f"{p}:{i}")
    assert not hits, f"potential secrets found: {hits[:5]}"
