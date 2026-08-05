"""Static validation of the Colab bundle: notebook structure, referenced files,
drive_cache imports, restore-flag ordering, and packaging rules."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "colab" / "HaluRISC_Training.ipynb"

REQUIRED_MARKERS = [
    "# 5b)", "# 6)", "# 6b)", "# 7.0)", "# 7)", "# 7b.0)", "# 7b)", "# 7b.5)",
    "# 7d)", "# 7d.5)", "# 7d.6)", "# 7e)", "# 7g.0)", "# 7g)", "# 7i)",
    "# 8.0)", "# 12.5)", "# 13)", "# 15)",
]

FLAGS = ["DRIVE_DIR", "CACHE_OK", "B2_OK", "B3_CACHE_OK", "B3_OK", "B4_OK", "VA_OK", "LEGACY_OK"]


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb["cells"]


def test_notebook_loads_and_has_all_markers():
    cells = _cells()
    assert len(cells) >= 35
    firsts = [
        "".join(c.get("source", [])).strip().splitlines()[0]
        for c in cells
        if "".join(c.get("source", [])).strip()
    ]
    for marker in REQUIRED_MARKERS:
        assert any(f.startswith(marker) for f in firsts), f"missing cell {marker}"


def test_every_python_invocation_points_to_existing_file():
    cells = _cells()
    pattern = re.compile(r"python (src/[\w/]+\.py)")
    found = 0
    for c in cells:
        src = "".join(c.get("source", []))
        for m in pattern.findall(src):
            assert (ROOT / m).exists(), f"{m} referenced but not in repo"
            found += 1
    assert found >= 10


def test_drive_cache_imports_resolve():
    from colab import drive_cache

    cells = _cells()
    for c in cells:
        src = "".join(c.get("source", []))
        for m in re.findall(r"from colab\.drive_cache import ([\w, ]+)", src):
            for name in m.replace(" ", "").split(","):
                assert hasattr(drive_cache, name), f"{name} missing from drive_cache"


def test_restore_flags_defined_before_use():
    """Every flag used in a run cell must be defined by an earlier restore cell."""
    cells = _cells()
    defined = set()
    for c in cells:
        src = "".join(c.get("source", []))
        for flag in FLAGS:
            if re.search(rf"^{flag}\s*=", src, re.M):
                defined.add(flag)
        for flag in FLAGS:
            if re.search(rf"\b{flag}\b", src) and flag not in defined:
                # DRIVE_DIR is set by cell 1; tolerate only first-cell references
                if flag == "DRIVE_DIR":
                    continue
                raise AssertionError(f"{flag} used in cell before being defined:\n{src[:300]}")
    assert defined >= {"DRIVE_DIR", "CACHE_OK", "B2_OK", "B3_CACHE_OK", "B3_OK", "B4_OK"}


def test_final_package_excludes_external_cache():
    cells = _cells()
    pkg = next(c for c in cells if "".join(c.get("source", [])).strip().startswith("# 15)"))
    src = "".join(pkg["source"])
    assert "b3_external_features.parquet" not in src
    assert "unified_records" not in src


def test_legacy_cells_skip_when_outputs_exist():
    cells = _cells()
    for marker in ("# 8)", "# 9)", "# 10)", "# 11)", "# 12)"):
        c = next(c for c in cells if "".join(c.get("source", [])).strip().startswith(marker))
        assert "already present - skipping" in "".join(c["source"]), f"{marker} lacks skip guard"


def test_l4_batch_sizes():
    cells = _cells()
    for marker in ("# 6)", "# 7e)"):
        c = next(c for c in cells if "".join(c.get("source", [])).strip().startswith(marker))
        assert "--batch-size 512" in "".join(c["source"]), f"{marker} not tuned for L4"
