"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Sparkles } from "lucide-react";
import { Thread } from "@/components/assistant-ui/thread";
import { MlStatus } from "@/components/ml-status";

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

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] gap-4">
      <div className="glass-panel p-4 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-lg font-bold gradient-text flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-violet-600 dark:text-purple-400" /> 💬 Chat Mode — Conversational AI Risk Analyst
          </h1>
          <p className="text-xs text-muted-foreground">
            Streaming via /api/chat (Vercel AI SDK); risk tool runs the calibrated XGBoost backend
          </p>
        </div>
        <MlStatus />
      </div>

      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
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
