"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Sparkles } from "lucide-react";
import { Thread } from "@/components/assistant-ui/thread";

export default function ChatPage() {
  const runtime = useChatRuntime();

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] gap-4">
      <div className="glass-panel p-4 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-lg font-bold gradient-text flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" /> 💬 Chat Mode — Conversational AI Risk Analyst
          </h1>
          <p className="text-xs text-muted-foreground">
            Powered by assistant-ui, GPT 5.6 Luna &amp; calibrated XGBoost (streaming via /api/chat)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-mono text-emerald-400">Streaming Connected</span>
        </div>
      </div>

      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </div>
  );
}
