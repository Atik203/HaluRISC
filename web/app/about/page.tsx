"use client";

import React from "react";
import { Info, Layers, Cpu } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-2xl">
        <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
          <Info className="w-6 h-6 text-violet-600 dark:text-purple-400" /> About HaluRISC Framework
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          Calibrated, explainable hallucination risk prediction for black-box LLM outputs
        </p>
      </div>

      {/* Methodology Section */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h2 className="text-base font-bold flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-600 dark:text-indigo-400" /> Pipeline Architecture
        </h2>
        <p className="text-xs text-muted-foreground leading-relaxed">
          HaluRISC evaluates candidate LLM answers against provided context without inspecting model weights or activations.
          Features are computed across seven feature groups: length/style, lexical overlap, entity coverage, numeric consistency, hedging density, NLI contradiction, and semantic embeddings.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-purple-700 dark:text-purple-300 mb-1">1. Feature Extraction</h4>
            <p className="text-xs text-muted-foreground">26 engineered features across 7 groups measuring grounding, entity coverage, NLI consistency, numeric novelty, and semantic drift.</p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300 mb-1">2. Calibrated XGBoost</h4>
            <p className="text-xs text-muted-foreground">XGBoost classifier with Platt scaling for trustworthy calibrated risk probabilities.</p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-blue-700 dark:text-blue-300 mb-1">3. SHAP Explanations</h4>
            <p className="text-xs text-muted-foreground">TreeExplainer attributions identifying specific features raising or lowering risk.</p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-1">4. Conversational AI UI</h4>
            <p className="text-xs text-muted-foreground">Next.js + assistant-ui + GPT 5.6 Luna for natural-language explanations with Generative UI.</p>
          </div>
        </div>
      </div>

      {/* Tech Stack List */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h2 className="text-base font-bold flex items-center gap-2">
          <Cpu className="w-5 h-5 text-violet-600 dark:text-purple-400" /> Core Tech Stack
        </h2>
        <div className="flex flex-wrap gap-2 text-xs">
          {["Next.js App Router", "assistant-ui", "Vercel AI SDK", "GPT 5.6 Luna", "FastAPI", "scikit-learn 1.9.0", "XGBoost 3.4.0", "PyTorch 2.11.0+cu128", "SHAP", "Recharts", "Tailwind CSS v4"].map((tech, i) => (
            <span key={i} className="bg-secondary px-3 py-1.5 rounded-lg border border-border text-foreground font-mono">
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
