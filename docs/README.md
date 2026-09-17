# HaluRISC Version B — Research Protocol Docs

Guides for the Version B phases (B5–B7). All procedures assume the repo root
with the `.venv` Python environment; paths are repo-root-relative.

| Doc | Covers |
|---|---|
| [b5-explanation-reliability.md](b5-explanation-reliability.md) | B5 experiments, outputs, and the **manual reviewer guide** (B5.5) |
| [b6-reproducibility.md](b6-reproducibility.md) | `run_all_experiments.py`, `manifest.json`, Docker — regenerate everything |
| [b7-research-ui.md](b7-research-ui.md) | Web routes, dashboard tabs, API contract, offline demo, deployment |

Quick reference:

- Experiment runner: `python src/models/run_all_experiments.py --dry-run`
- Artifact gate: `python src/models/verify_artifacts.py`
- Manifest freeze: `python src/models/make_manifest.py`
- Tests: `python -m pytest tests`
- Web: `python -m uvicorn src.api.main:app --port 8000` then `pnpm dev` in `web/`
