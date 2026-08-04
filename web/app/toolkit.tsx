"use generative";

import { defineToolkit, externalTool } from "@assistant-ui/react";
import { z } from "zod";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";

interface Prediction {
  calibrated_score: number;
  label: string;
  latency_ms?: number;
  model_version?: string;
  error?: string;
  warning?: string;
}

interface Explanation {
  top_features?: Array<{ feature: string; value: number; impact: number }>;
  base_value?: number;
}

interface AnalyzeResult {
  prediction: Prediction;
  explanation?: Explanation | null;
}

interface AnalyzeRenderProps {
  result?: AnalyzeResult;
  status: { type: string; reason?: string; error?: unknown };
}

function RiskBanner({ prediction }: { prediction: Prediction }) {
  const score = Number(prediction.calibrated_score ?? 0);
  const pct = Math.min(100, Math.max(0, Math.round(score * 100)));

  let emoji = "🟢";
  let labelText = "Low Risk";
  let toneClass = "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400";
  if (pct >= 70) {
    emoji = "🔴";
    labelText = "High Risk";
    toneClass = "border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-400";
  } else if (pct >= 30) {
    emoji = "🟡";
    labelText = "Medium Risk";
    toneClass = "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400";
  }

  return (
    <div className={`flex items-center justify-between gap-3 rounded-xl border px-4 py-3 ${toneClass}`}>
      <div className="flex items-center gap-3">
        <span className="text-3xl leading-none">{emoji}</span>
        <div>
          <div className="text-sm font-bold">{labelText}</div>
          <div className="text-xs opacity-80">{pct}% calibrated risk probability</div>
        </div>
      </div>
      <div className="flex flex-col items-end gap-1 text-[10px] font-mono">
        {prediction.latency_ms != null && <span>⚡ {prediction.latency_ms} ms</span>}
        {prediction.model_version && <span>model {prediction.model_version}</span>}
      </div>
    </div>
  );
}

export default defineToolkit({
  analyze_hallucination: {
    description: "Analyze an answer for hallucination risk",
    parameters: z.object({
      question: z.string(),
      context: z.string(),
      answer: z.string(),
    }),
    execute: externalTool(),
    render: ({ result, status }: AnalyzeRenderProps) => {
      if (status?.type === "running") {
        return (
          <div className="glass-panel flex flex-col gap-3 rounded-xl p-4 animate-pulse">
            <div className="h-6 w-32 bg-muted rounded-md" />
            <div className="h-24 w-full bg-muted rounded-md" />
          </div>
        );
      }

      if (!result || !result.prediction || result.prediction.error) {
        return (
          <div className="glass-panel rounded-xl p-4 text-xs text-muted-foreground">
            {result?.prediction?.error || "ML analysis unavailable."} Make sure the FastAPI backend is running on port 8000.
          </div>
        );
      }

      const { prediction, explanation } = result;

      return (
        <div className="flex w-full flex-col gap-4 my-3">
          <RiskBanner prediction={prediction} />
          {prediction.warning && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-700 dark:text-amber-400">
              ⚠️ {prediction.warning}
            </p>
          )}
          <RiskGauge
            score={prediction.calibrated_score}
            label={prediction.label}
            latencyMs={prediction.latency_ms}
          />
          {explanation && explanation.top_features && (
            <ShapChart
              features={explanation.top_features}
              baseValue={explanation.base_value}
            />
          )}
        </div>
      );
    },
  },
});
