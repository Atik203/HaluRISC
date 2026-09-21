"use client";

import React from "react";
import { Info } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

interface FeatureImpact {
  feature: string;
  value: number;
  impact: number; // Positive = increases risk, Negative = decreases risk
}

interface ShapChartProps {
  features: FeatureImpact[];
  baseValue?: number;
}

// Plain-language map for feature names
const FEATURE_NAME_MAP: Record<string, string> = {
  overlap_answer_context: "Context Overlap",
  overlap_answer_question: "Question Overlap",
  jaccard_ans_ctx: "Jaccard Similarity",
  jaccard_ans_q: "Question Jaccard",
  n_numbers_answer: "Answer Numbers Count",
  n_numbers_context: "Context Numbers Count",
  number_overlap_ratio: "Number Overlap Ratio",
  novel_numbers: "Novel Numbers Found",
  hedge_count: "Hedge Phrase Count",
  hedge_density: "Hedge Density",
  n_words: "Answer Word Count",
  n_chars: "Answer Character Count",
  n_sentences: "Sentence Count",
  avg_word_len: "Avg Word Length",
};

export function ShapChart({ features, baseValue }: ShapChartProps) {
  const data = features.map((f) => ({
    name: FEATURE_NAME_MAP[f.feature] || f.feature,
    impact: Number(f.impact.toFixed(4)),
    rawName: f.feature,
    rawValue: f.value,
  }));

  return (
    <div className="w-full glass-panel p-5 rounded-2xl">
      <div className="flex justify-between items-start gap-3 mb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-wide">SHAP feature contributions</h3>
          <p className="text-xs text-muted-foreground">
            Red pushed the score up · Green pulled it down
          </p>
        </div>
        {baseValue !== undefined && (
          <span
            title="The score the model would give if no feature spoke for or against the answer."
            className="shrink-0 text-xs text-muted-foreground bg-secondary px-2.5 py-1 rounded-md"
          >
            Model average {baseValue.toFixed(2)}
          </span>
        )}
      </div>

      <details className="group mb-4">
        <summary className="flex cursor-pointer items-center gap-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground">
          <Info className="h-3.5 w-3.5" aria-hidden /> How to read this chart
        </summary>
        <ul className="mt-2 space-y-1 text-[11px] leading-relaxed text-muted-foreground">
          <li>
            Each bar is one measured feature of this answer. The bar length is how much that feature moved the raw
            model output away from the model average.
          </li>
          <li>
            Red bars raised the hallucination risk. Green bars lowered it. Only the largest contributions are shown,
            so small effects are hidden.
          </li>
          <li>
            SHAP explains how the model reasoned about this answer. It is not proof that the answer is true or false.
            The calibrated score is the number the system reports, and the two can disagree.
          </li>
        </ul>
      </details>

      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 20 }}
          >
            <XAxis
              type="number"
              stroke="var(--muted-foreground)"
              fontSize={11}
              label={{
                value: "SHAP value (impact on the raw risk score)",
                fontSize: 10,
                position: "insideBottom",
                offset: -8,
                fill: "var(--muted-foreground)",
              }}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="var(--muted-foreground)"
              fontSize={11}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                borderColor: "var(--border)",
                color: "var(--popover-foreground)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(val: number | string) => [`Impact: ${val}`, "SHAP Value"]}
            />
            <ReferenceLine x={0} stroke="var(--muted-foreground)" strokeOpacity={0.4} strokeDasharray="3 3" />
            <Bar dataKey="impact" radius={[4, 4, 4, 4]} isAnimationActive={false}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.impact >= 0 ? "var(--risk-high)" : "var(--risk-low)"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
