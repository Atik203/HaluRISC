# HaluRISC — AI Agent Guidelines & Optimization Protocol (.agents/AGENTS.md)

See project root AGENTS.md for full details.

## Quick Reference Rules:
1. **Prompt Caching:** Place static system prompts (>1024 tokens) FIRST in message arrays to maximize prompt cache hits on DeepSeek v4 Flash ($0.0028/M cached vs $0.14/M) and GPT 5.6 Luna ($0.02/M cached vs $0.20/M).
2. **assistant-ui Documentation:** Follow https://www.assistant-ui.com/docs. Use modern `defineToolkit` + `"use generative"` directive.
3. **Open-Code Customization:** `assistant-ui` components live in `components/assistant-ui/` (shadcn-style). Edit them directly in the repo.
4. **Architecture:** Next.js App Router (BFF on port 3000) proxies `/api/ml/*` to FastAPI (Python ML on port 8000).
