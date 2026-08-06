"""
Regenerate the SELF-CONTAINED Colab notebook (no zip upload needed).

Embeds every runtime source file (src/**/*.py, colab/drive_cache.py,
colab/requirements-colab.txt) as base64 inside notebook cell 3. In Colab,
running cell 3 writes all files to /content/HaluRISC/ and verifies their
sha256 hashes, so the user uploads ONLY the single .ipynb file.

Workflow after any source change:
  python colab/build_self_contained.py     # embeds current src into cell 3
  # user re-uploads just colab/HaluRISC_Training.ipynb

To patch a single file in an existing session: paste a small cell that
rewrites just that file (or re-run cell 3 after editing its EMBEDDED entry).
"""

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "HaluRISC_Training.ipynb"

EMBED_PATHS = [
    *sorted((ROOT / "src").rglob("*.py")),
    ROOT / "colab" / "drive_cache.py",
    ROOT / "colab" / "requirements-colab.txt",
]
EXCLUDE_DIRS = {"__pycache__"}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_cell(embedded: dict, hashes: dict) -> dict:
    cell = (
        "# 3) SELF-CONTAINED: write the HaluRISC source from this cell (NO zip upload)\n"
        "# Regenerate with: python colab/build_self_contained.py\n"
        "# To patch a single file later: edit its EMBEDDED entry below and rerun\n"
        "# this cell, or paste a small cell that rewrites just that file.\n"
        "import base64, hashlib, os\n"
        "\n"
        "ROOT = '/content/HaluRISC'\n"
        "os.makedirs(ROOT, exist_ok=True)\n"
        "\n"
        f"EMBEDDED = {json.dumps(embedded, indent=1)}\n"
        f"HASHES = {json.dumps(hashes, indent=1)}\n"
        "\n"
        "for rel, b64 in EMBEDDED.items():\n"
        "    dest = os.path.join(ROOT, rel)\n"
        "    os.makedirs(os.path.dirname(dest), exist_ok=True)\n"
        "    with open(dest, 'wb') as f:\n"
        "        f.write(base64.b64decode(b64))\n"
        "\n"
        "bad = [rel for rel in EMBEDDED\n"
        "       if hashlib.sha256(open(os.path.join(ROOT, rel), 'rb').read()).hexdigest() != HASHES[rel]]\n"
        "assert not bad, f'embedded source mismatch: {bad}'\n"
        "os.chdir(ROOT)\n"
        "print(f'Self-contained source ready: {len(EMBEDDED)} files in {ROOT}')\n"
        "print('src present:', os.path.exists(os.path.join(ROOT, 'src')))\n"
    )
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": cell.splitlines(keepends=True)}


def main():
    embedded = {}
    hashes = {}
    for path in EMBED_PATHS:
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            continue
        rel = path.relative_to(ROOT).as_posix()
        embedded[rel] = b64(path)
        hashes[rel] = sha256_bytes(path.read_bytes())

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    new_cell = build_cell(embedded, hashes)

    replaced = False
    for i, cell in enumerate(notebook["cells"]):
        src = "".join(cell.get("source", []))
        first = src.strip().splitlines()[0] if src.strip() else ""
        if first.startswith("# 3)"):
            notebook["cells"][i] = new_cell
            replaced = True
            break
    assert replaced, "cell 3 not found"

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    size_kb = NOTEBOOK.stat().st_size / 1024
    print(f"Embedded {len(embedded)} files into cell 3; notebook size {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
