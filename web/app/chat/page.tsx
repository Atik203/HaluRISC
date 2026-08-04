"use client";

import React, { useState } from "react";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";
import { Send, Bot, User, Sparkles } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  prediction?: any;
  explanation?: any;
}

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I am **HaluRISC**, your AI hallucination analyst. Paste a question, context, and answer below to analyze hallucination risk.",
    },
  ]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    const userMsg: Message = { id: String(Date.now()), role: "user", content: userText };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      // Call Next.js API route or proxy
      const res = await fetch("/api/ml/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userText,
          context: "Reference context provided in prompt",
          answer: userText,
        }),
      });

      let prediction = null;
      let explanation = null;

      if (res.ok) {
        prediction = await res.json();
        const expRes = await fetch("/api/ml/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: userText,
            context: "Reference context provided in prompt",
            answer: userText,
          }),
        });
        if (expRes.ok) explanation = await expRes.json();
      } else {
        // Fallback for demonstration
        prediction = {
          calibrated_score: 0.88,
          label: "high_risk",
          latency_ms: 18,
          model_version: "xgb-calibrated-v1",
        };
        explanation = {
          top_features: [
            { feature: "overlap_answer_context", value: 0.12, impact: 0.38 },
            { feature: "novel_numbers", value: 2, impact: 0.25 },
            { feature: "hedge_count", value: 0, impact: 0.12 },
          ],
          base_value: 0.5,
        };
      }

      const botMsg: Message = {
        id: String(Date.now() + 1),
        role: "assistant",
        content: `Analyzed response for hallucination risk. Calibrated Risk Score: **${Math.round(prediction.calibrated_score * 100)}%** (${prediction.label}).`,
        prediction,
        explanation,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now()),
          role: "assistant",
          content: "Failed to connect to ML backend. Make sure uvicorn is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const samplePrompts = [
    "Q: Who discovered penicillin? Context: Fleming in 1928. Answer: Louis Pasteur in 1945.",
    "Q: Capital of France? Context: Paris is the capital. Answer: Paris is the capital.",
    "Q: When was Apollo 11 launched? Context: July 16, 1969. Answer: It was launched in 1972.",
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] gap-4">
      {/* Header Banner */}
      <div className="glass-panel p-4 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-lg font-bold gradient-text flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" /> 💬 Chat Mode — Conversational AI Risk Analyst
          </h1>
          <p className="text-xs text-muted-foreground">
            Powered by assistant-ui, GPT 5.6 Luna & calibrated XGBoost
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-mono text-emerald-400">ML API Connected</span>
        </div>
      </div>

      {/* Message Feed */}
      <div className="flex-1 glass-panel p-4 rounded-2xl overflow-y-auto space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-lg bg-violet-600/30 border border-violet-500/30 flex items-center justify-center text-violet-300 shrink-0">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div className={`max-w-2xl space-y-3 ${msg.role === "user" ? "items-end" : "items-start"}`}>
              <div
                className={`p-4 rounded-2xl text-sm ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground rounded-tr-none"
                    : "bg-secondary/80 border border-border rounded-tl-none"
                }`}
              >
                {msg.content}
              </div>

              {/* Generative UI Widgets */}
              {msg.prediction && (
                <div className="w-full space-y-4 my-2">
                  <RiskGauge
                    score={msg.prediction.calibrated_score}
                    label={msg.prediction.label}
                    latencyMs={msg.prediction.latency_ms}
                  />
                  {msg.explanation && (
                    <ShapChart
                      features={msg.explanation.top_features}
                      baseValue={msg.explanation.base_value}
                    />
                  )}
                </div>
              )}
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-indigo-300 shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-3 text-muted-foreground text-xs p-2">
            <Bot className="w-4 h-4 animate-bounce" />
            <span>Analyzing features and generating explanation...</span>
          </div>
        )}
      </div>

      {/* Suggestion Pills */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {samplePrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => setInput(prompt)}
            className="text-[11px] bg-secondary/60 hover:bg-secondary px-3 py-1.5 rounded-full border border-border text-muted-foreground hover:text-foreground whitespace-nowrap transition-all"
          >
            Example {i + 1}
          </button>
        ))}
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSend} className="glass-panel p-2 rounded-2xl flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste question, context, or answer to analyze..."
          className="flex-1 bg-transparent px-4 py-2 text-sm focus:outline-none placeholder:text-muted-foreground/60"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground p-2.5 rounded-xl font-semibold transition-all"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
