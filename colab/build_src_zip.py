"""
Regenerate colab/halurisc_src.zip (roadmap B6.3: rebuild after every source
change used by Colab).

Includes: src/, tests/, requirements.txt, colab/HaluRISC_Training.ipynb,
colab/requirements-colab.txt. Excludes __pycache__, *.pyc, .ipynb_checkpoints,
and the zip itself, so the bundle stays clean and Windows-free.

Run (repo root):
  python colab/build_src_zip.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "halurisc_src.zip"
PARTS = [
    ROOT / "src",
    ROOT / "tests",
    ROOT / "requirements.txt",
    ROOT / "colab" / "HaluRISC_Training.ipynb",
    ROOT / "colab" / "requirements-colab.txt",
]
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for part in PARTS:
            if part.is_dir():
                for f in sorted(part.rglob("*")):
                    if not f.is_file():
                        continue
                    if any(ex in f.parts for ex in EXCLUDE_DIRS) or f.suffix == ".pyc":
                        continue
                    z.write(f, f.relative_to(ROOT))
                    n += 1
            elif part.is_file():
                z.write(part, part.relative_to(ROOT))
                n += 1
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT} ({size_mb:.2f} MB, {n} files, no __pycache__/.pyc)")


if __name__ == "__main__":
    main()
