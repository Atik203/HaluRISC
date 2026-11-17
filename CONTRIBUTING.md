# Contributing

Thanks for taking an interest in HaluRISC. This repository contains a research
artifact: a feature-based hallucination risk scorer, the evaluation code, the
FastAPI service, and the Next.js dashboard.

## Ways to contribute

- Report a bug or a reproducibility problem.
- Propose a feature or an evaluation setting.
- Improve documentation or tests.
- Send a pull request with a focused fix.

Please open an issue before starting a large change, so the scope can be agreed
first. For security problems, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## Development setup

### Python pipeline and API

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests -v
```

The API runs on `http://127.0.0.1:8000`:

```bash
uvicorn src.api.main:app --port 8000
```

### Web dashboard

```bash
cd web
pnpm install
pnpm run dev      # http://localhost:3000
pnpm run lint
pnpm run build
```

Use `pnpm`, not `npm`, because the lockfile is `pnpm-lock.yaml`.

## Pull request checklist

- The change is limited to one topic.
- `pytest tests` passes.
- `pnpm run lint` and `pnpm run build` pass for web changes.
- New behaviour has a test or a documented reason why it does not.
- Reported numbers come from artifacts, never from memory.
- No secrets, `.env` files, raw datasets, or personal data are committed.

## Commit messages

Use a short imperative subject line, for example:

```
Add length-matched external control
Fix calibration table rounding
```

## Style

- Python: standard library formatting, type hints where practical, no wildcard imports.
- TypeScript: the existing ESLint configuration, two-space indentation.
- LaTeX: keep the manuscript sources private; do not add paper files to the repository.
