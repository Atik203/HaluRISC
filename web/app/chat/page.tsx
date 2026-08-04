"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Sparkles } from "lucide-react";
import { Thread } from "@/components/assistant-ui/thread";

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
            Powered by assistant-ui, GPT 5.6 Luna &amp; calibrated XGBoost (streaming via /api/chat)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400">Streaming Connected</span>
        </div>
      </div>

      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </div>
  );
}
