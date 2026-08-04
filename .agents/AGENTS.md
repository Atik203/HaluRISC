# HaluRISC — AI Agent Guidelines & Optimization Protocol (.agents/AGENTS.md)

See project root AGENTS.md for the full binding rules.

## Quick Reference Rules:
1. **Source of Truth:** `blueprint.md` (Version A course scope only) + `roadmap.md` (guidance; version pins may be stale — use latest stable installed versions).
2. **Prompt Caching:** Place static system prompts (>1024 tokens) FIRST in message arrays to maximize prompt cache hits on DeepSeek v4 Flash ($0.0028/M cached vs $0.14/M) and GPT 5.6 Luna ($0.02/M cached vs $0.20/M).
3. **assistant-ui Documentation:** Follow https://www.assistant-ui.com/docs. Use modern `defineToolkit` + `"use generative"` directive; chat MUST use Thread primitives + `/api/chat`.
4. **Open-Code Customization:** `assistant-ui` components live in `components/assistant-ui/`. Edit them directly in the repo.
5. **Architecture:** Next.js App Router (BFF on port 3000) proxies `/api/ml/*` to FastAPI (Python ML on port 8000). API loads artifacts; never trains.
6. **ML Protocol:** 5-fold CV tuning, 3 seeds (42/123/456) mean±std, calibration fit on validation only, McNemar + bootstrap CIs, group ablations, saved split indices, pinned `requirements.txt` (==).
7. **Environment Secrets:** Copy `.env.example` to `web/.env.local` (Next.js, `OPENAI_API_KEY`) and `.env` (Python backend). Never commit real keys.
8. **No fabricated data:** dashboard numbers must come from `artifacts/results/*`.
