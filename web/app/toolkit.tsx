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
    execute: externalTool(),
    render: ({ result, status }: any) => {
      if (status?.type === "running") {
        return (
          <div className="p-4 rounded-xl glass-panel animate-pulse flex flex-col gap-3">
            <div className="h-6 w-32 bg-muted rounded-md" />
            <div className="h-24 w-full bg-muted rounded-md" />
          </div>
        );
      }

      if (!result || !result.prediction) return null;

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
