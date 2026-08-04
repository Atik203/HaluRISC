"use client";

import React from "react";
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
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-sm font-semibold tracking-wide">SHAP Feature Contributions</h3>
          <p className="text-xs text-muted-foreground">
            Red increases hallucination risk · Green decreases risk
          </p>
        </div>
        {baseValue !== undefined && (
          <span className="text-xs text-muted-foreground bg-secondary px-2.5 py-1 rounded-md">
            Base value: {baseValue.toFixed(2)}
          </span>
        )}
      </div>

      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <XAxis type="number" stroke="var(--muted-foreground)" fontSize={11} />
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
            <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.impact >= 0 ? "#ef4444" : "#22c55e"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
