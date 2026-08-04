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
        <div className="flex flex-col gap-4 my-3 w-full">
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
