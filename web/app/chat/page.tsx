"use client";

import { useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { FileText, Sparkles, X } from "lucide-react";
import { Thread } from "@/components/assistant-ui/thread";
import { MlStatus } from "@/components/ml-status";
import { AutoAnalysisProvider } from "@/components/assistant-ui/auto-analysis-context";
import { CONTEXT_MAX } from "@/lib/analysis-input";

const SUGGESTIONS = [
  {
    title: "Hallucinated answer",
    label: "🔴 Check a hallucinated answer",
    prompt:
      "Check this answer for hallucination risk. Question: 'What is the capital of France?' Context: 'France is a country in Europe. Its capital city is Paris.' Answer: 'The capital of France is Lyon, and it has been since 1800.'",
  },
  {
    title: "Grounded answer",
    label: "🟢 Check a grounded answer",
    prompt:
      "Check this answer for hallucination risk. Question: 'Who discovered penicillin?' Context: 'Penicillin was discovered by Alexander Fleming in 1928.' Answer: 'Penicillin was discovered by Alexander Fleming.'",
  },
  {
    title: "How it works",
    label: "❓ How does HaluRISC work?",
    prompt: "Explain how HaluRISC detects hallucinations and how to interpret the risk score.",
  },
  {
    title: "Borderline case",
    label: "⚖️ Borderline example",
    prompt:
      "Give me a borderline example where the risk score would be around 50% and explain why calibration matters.",
  },
];

export default function ChatPage() {
  const runtime = useChatRuntime({ suggestions: SUGGESTIONS });
  const [evidence, setEvidence] = useState("");
  const [autoEnabled, setAutoEnabled] = useState(true);

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] gap-4">
      <div className="glass-panel p-4 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-lg font-bold gradient-text flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-violet-600 dark:text-purple-400" /> 💬 Chat Mode — Conversational AI Risk Analyst
          </h1>
          <p className="text-xs text-muted-foreground">
            Streaming via /api/chat (Vercel AI SDK); every answer is auto-checked by the calibrated XGBoost backend
          </p>
        </div>
        <MlStatus />
      </div>

      {/* Evidence context + auto-analysis controls */}
      <div className="glass-panel rounded-2xl p-3 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-xs font-semibold text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoEnabled}
              onChange={(e) => setAutoEnabled(e.target.checked)}
              className="accent-violet-600 w-4 h-4"
            />
            Auto risk check (per answer)
          </label>
          <span className="text-[10px] text-muted-foreground font-mono">
            {evidence ? `${evidence.length.toLocaleString()} / ${CONTEXT_MAX.toLocaleString()} chars` : "no evidence set — conversation-only grounding"}
          </span>
        </div>
        <details className="group">
          <summary className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-muted-foreground hover:text-foreground">
            <FileText className="w-4 h-4 text-violet-600 dark:text-purple-400" aria-hidden />
            Evidence context (optional) — answers are checked against this
          </summary>
          <div className="mt-2 flex flex-col sm:flex-row gap-2">
            <textarea
              value={evidence}
              onChange={(e) => setEvidence(e.target.value.slice(0, CONTEXT_MAX))}
              placeholder="Paste a document, article, or reference material once — then just chat. Every answer is automatically scored against this evidence."
              rows={3}
              className="flex-1 bg-secondary/50 border border-border rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setEvidence("")}
                aria-label="Clear evidence context"
                className="text-xs font-semibold bg-secondary/80 hover:bg-secondary border border-border px-3 py-2 rounded-xl transition-all inline-flex items-center gap-1"
              >
                <X className="w-3.5 h-3.5" aria-hidden /> Clear
              </button>
            </div>
          </div>
        </details>
      </div>

      <AssistantRuntimeProvider runtime={runtime}>
        <AutoAnalysisProvider config={{ enabled: autoEnabled, sessionContext: evidence.trim() || null }}>
          <Thread />
        </AutoAnalysisProvider>
      </AssistantRuntimeProvider>

      <p className="text-[10px] text-muted-foreground text-center">
        Chat streams via /api/chat and needs <code className="font-mono">OPENAI_API_KEY</code> in{" "}
        <code className="font-mono">web/.env.local</code>. Without it, the model-risk tool still runs — use{" "}
        <a href="/analyze" className="underline text-violet-600 dark:text-purple-400">Analyze</a> or the{" "}
        <a href="/demo" className="underline text-violet-600 dark:text-purple-400">offline demo</a>.
      </p>
    </div>
  );
}
