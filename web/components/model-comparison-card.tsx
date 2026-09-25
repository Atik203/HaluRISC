"use client";

import React, { useEffect, useState } from "react";
import { BarChart3, RefreshCw, TriangleAlert } from "lucide-react";

interface CompareModelScore {
  score: number;
  label: string;
  decision_threshold: number;
}

interface CompareResponse {
  models: Record<string, CompareModelScore>;
  deployed: {
    calibrated_score: number;
    risk_score: number;
    legacy_score: number;
    label: string;
  };
  thresholds?: { low?: number; medium?: number };
  latency_ms: number;
}

export interface ModelComparisonInput {
  question: string;
  context: string;
  answer: string;
}

const MODEL_ORDER: Array<[string, string]> = [
  ["xgboost", "XGBoost (raw)"],
  ["random_forest", "Random forest"],
  ["logistic_regression", "Logistic regression"],
  ["heuristic_overlap", "Overlap heuristic"],
];

function barColor(label: string) {
  if (label === "high_risk") return "var(--risk-high)";
  if (label === "medium_risk") return "var(--risk-medium)";
  return "var(--risk-low)";
}

function formatScore(score: number) {
  if (score >= 0.999) return "99.9%";
  return `${(score * 100).toFixed(1)}%`;
}

function ScoreBar({
  score,
  label,
  markers = [],
}: {
  score: number;
  label: string;
  markers?: number[];
}) {
  return (
    <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-secondary/70" aria-hidden>
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${Math.max(score * 100, 0.5)}%`, background: barColor(label) }}
      />
      {markers.map((m) => (
        <span
          key={m}
          className="absolute top-0 h-full w-px bg-foreground/50"
          style={{ left: `${Math.min(m * 100, 100)}%` }}
        />
      ))}
    </div>
  );
}

interface CompareState {
  key: string;
  data?: CompareResponse;
  error?: string;
}

export function ModelComparisonCard({
  input,
  title = "Model comparison",
}: {
  input: ModelComparisonInput;
  title?: string;
}) {
  const { question, context, answer } = input;
  const requestKey = `${question}\u001f${context}\u001f${answer}`;
  const [state, setState] = useState<CompareState | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const key = `${question}\u001f${context}\u001f${answer}`;
    fetch("/api/ml/predict/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context, answer }),
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (r) => {
        if (!r.ok) {
          const detail = (await r.json().catch(() => null))?.detail;
          throw new Error(detail || `ML backend error (${r.status}).`);
        }
        return (await r.json()) as CompareResponse;
      })
      .then((d) => setState({ key, data: d }))
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setState({ key, error: err instanceof Error ? err.message : "Could not load model scores." });
      });
    return () => controller.abort();
  }, [question, context, answer]);

  const current = state?.key === requestKey ? state : null;
  const data = current?.data ?? null;
  const error = current?.error ?? null;
  const loading = current === null;

  const known = new Set(MODEL_ORDER.map(([key]) => key));
  const extras = data ? Object.keys(data.models).filter((k) => !known.has(k)) : [];

  return (
    <section className="glass-panel rounded-2xl p-5 space-y-4 animate-fade">
      <div className="flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-bold">
          <BarChart3 className="w-4 h-4 text-violet-600 dark:text-purple-400" aria-hidden /> {title}
        </h3>
        {data && (
          <span className="font-mono text-[10px] text-muted-foreground">{data.latency_ms.toFixed(0)} ms</span>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <RefreshCw className="w-3.5 h-3.5 animate-spin text-violet-500" aria-hidden />
          Scoring with every model…
        </div>
      )}

      {error && (
        <p className="flex items-start gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {error}
        </p>
      )}

      {data && (
        <div className="space-y-3">
          <div className="space-y-1.5">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-xs font-semibold">Deployed calibrated score</span>
              <span className="font-mono tnum text-xs font-semibold">
                {formatScore(data.deployed.calibrated_score)}
              </span>
            </div>
            <ScoreBar
              score={data.deployed.calibrated_score}
              label={data.deployed.label}
              markers={[data.thresholds?.low ?? 0.35, data.thresholds?.medium ?? 0.6]}
            />
          </div>

          <div className="border-t border-border/50 pt-3 space-y-2.5">
            <p className="eyebrow text-muted-foreground">B2 baselines · raw probabilities</p>
            {[...MODEL_ORDER, ...extras.map((k) => [k, k] as [string, string])].map(([key, label]) => {
              const entry = data.models[key];
              if (!entry) return null;
              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-xs">{label}</span>
                    <span className="font-mono tnum text-xs">{formatScore(entry.score)}</span>
                  </div>
                  <ScoreBar score={entry.score} label={entry.label} markers={[entry.decision_threshold]} />
                </div>
              );
            })}
          </div>

          <p className="text-[10px] leading-relaxed text-muted-foreground">
            Bars are raw model probabilities from the saved B2 seed-42 artifacts. The tick on each bar is that model&apos;s
            decision rule (0.5 for the learned models, 0.03 for the overlap heuristic). The highlighted row is the
            calibrated score the deployed system shows, with band markers at 0.35 and 0.60.
          </p>
        </div>
      )}
    </section>
  );
}
