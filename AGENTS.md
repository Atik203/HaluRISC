# HaluRISC — AI Agent Guidelines & Optimization Protocol (AGENTS.md)

This document specifies mandatory rules, design patterns, prompt caching optimizations, and documentation references for AI agents (and developers) working on the HaluRISC codebase.

---

## 1. LLM Prompt Caching Maximization Protocol

To minimize API latency and reduce LLM token costs by up to **90–98%** (e.g., DeepSeek v4 Flash cache hit at $0.0028/M tokens vs $0.14/M; GPT 5.6 Luna cached input at $0.02/M vs $0.20/M), all LLM integration code MUST adhere to the following rules:

### 1.1 Static System Prompt Prefixing
- **Rule:** Place all static instructions, system definitions, persona rules, and tool schemas at the **very beginning** of the prompt or message array.
- **Rule:** Never insert dynamic variables (timestamps, user IDs, query-specific data) into the middle of the system prompt.
- **Minimum Prefix Threshold:** Keep the static system prefix above 1,024 tokens to guarantee prefix cache hits on DeepSeek v4 Flash and OpenAI models.

```typescript
// ✅ CORRECT: Static prefix remains 100% identical across requests -> High Cache Hit Rate
const SYSTEM_PROMPT = `
You are HaluRISC, an AI hallucination risk analyst... [Static Instructions > 1024 tokens]
Tool Definitions: [Static Tool Schemas]
Rules: [Static Rules]
`;

const messages = [
  { role: "system", content: SYSTEM_PROMPT },
  ...userConversationHistory, // Dynamic user inputs placed strictly AFTER static system prompt
];

// ❌ INCORRECT: Dynamic variables inside system prompt destroy context caching
const BAD_SYSTEM_PROMPT = `You are HaluRISC. Time: ${new Date().toISOString()}. User: ${userId}...`;
```

### 1.2 Tool Definition Standardization
- Standardize tool definitions across calls using `zod` schemas.
- Do not dynamically generate tool schemas per request; export a static `tools` object.

### 1.3 Target API Benchmarks & Caching Multipliers
- **DeepSeek-V4-Flash-0731:** $0.14/M input (cache miss) vs **$0.0028/M input (cache hit — 50x discount)**.
- **GPT-5.6-Luna:** $0.20/M input (cache miss) vs **$0.02/M input (cache hit — 10x discount)**.

---

## 2. assistant-ui Documentation & Integration Map

When building or modifying the frontend chat interface, refer to these canonical documentation sources and architectural rules:

### 2.1 Official Documentation Reference
- **Primary Docs:** [https://www.assistant-ui.com/docs](https://www.assistant-ui.com/docs)
- **Component Registry:** `https://r.assistant-ui.com/styles/default/{name}.json`
- **CLI Scaffolding:** `npx assistant-ui@latest create web`
- **Component Installation:** `npx shadcn@latest add @assistant-ui/thread`

### 2.2 Core Runtime Stack
- **Next.js Integration Adapter:** `@assistant-ui/react-ai-sdk` via `useChatRuntime` hook.
- **Streaming Provider:** Vercel AI SDK (`ai` package) using `streamText()` and `toDataStreamResponse()`.
- **API Route Location:** `app/api/chat/route.ts` (Next.js App Router).

### 2.3 Generative UI & Toolkits (Modern API)
- Use `defineToolkit` with the `"use generative"` directive in `app/toolkit.tsx`.
- **Do NOT use deprecated APIs** (`makeAssistantToolUI`, `makeAssistantTool`).
- Wrap Next.js config with `withAui()` plugin from `@assistant-ui/next`.

```tsx
// app/toolkit.tsx — Modern Generative UI Pattern
"use generative";
import { defineToolkit, externalTool } from "@assistant-ui/react";
import { z } from "zod";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";

export default defineToolkit({
  analyze_hallucination: {
    description: "Analyze an answer for hallucination risk",
    parameters: z.object({
      question: z.string(),
      context: z.string(),
      answer: z.string(),
    }),
    execute: externalTool(), // Executed server-side in /api/chat route
    render: ({ result, status }) => {
      if (status.type === "running") return <LoadingSkeleton />;
      return (
        <div className="rounded-lg border bg-card p-4">
          <RiskGauge score={result.prediction.calibrated_score} label={result.prediction.label} />
          <ShapChart features={result.explanation.top_features} baseValue={result.explanation.base_value} />
        </div>
      );
    },
  },
});
```

---

## 3. Open-Code Philosophy & LLM Assisting

assistant-ui follows the **shadcn/ui "Open Code" philosophy**:

1. **Source in Your Repo:** `assistant-ui` components live directly inside your `components/assistant-ui/` directory. They are NOT hidden inside `node_modules`.
2. **LLM Assistance Advantage:** Because the component source code is in your codebase, LLMs (Claude, Cursor, Copilot) can read, understand, and directly refactor the components, Tailwind classes, and Radix primitives without guessing abstract library APIs.
3. **Customization Rule:** Modify files in `components/assistant-ui/` directly to match the HaluRISC dark theme, blue-violet accent gradients, and typography without fear of breaking updates.

---

## 4. Codebase Architecture & Boundary Rules

- **Next.js (`web/`):** Serves as the Backend-for-Frontend (BFF). Handles UI rendering, static pages, OpenAI API key security, and streaming (`/api/chat`).
- **FastAPI (`src/api/`):** Pure Python ML inference server running on `http://127.0.0.1:8000`. Serves XGBoost predictions, NLI feature extraction, SHAP explanations, and LLM-as-judge runs (`/predict`, `/explain`, `/judge`).
- **Next.js Proxy:** All requests from frontend to FastAPI MUST route through `next.config.ts` rewrites (`/api/ml/:path*` → `http://127.0.0.1:8000/:path*`) to prevent CORS issues.
- **Python ML Pipeline:** Python code MUST use `.venv` (`python -m venv .venv`) with `numpy<2` compatibility for pandas/scipy/xgboost C-extensions.

---

## 5. Environment & Secrets Configuration (.env Placement)

- **Template Reference File:** Root [**.env.example**](file:///d:/ML/HaluRISC/.env.example) contains all environment variable keys and descriptions.
- **Next.js Frontend Environment:**
  - **Location:** `web/.env.local`
  - **Keys:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `NEXT_PUBLIC_ML_API_URL`
  - **Security Rule:** Server-only variables (`OPENAI_API_KEY`) must NEVER start with `NEXT_PUBLIC_`. They are strictly accessed in `app/api/chat/route.ts` (server side).
- **FastAPI Python Backend Environment:**
  - **Location:** Root `.env` or system environment variables loaded via `python-dotenv`.
  - **Keys:** `FASTAPI_HOST`, `FASTAPI_PORT`, `OPENAI_API_KEY` (for `/judge`), `DEEPSEEK_API_KEY`.
- **Git Security Rule:** Neither `.env` nor `.env.local` are ever committed to Git (`.gitignore` protects both).
