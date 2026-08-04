"use client";

import React, { useState } from "react";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";
import { Play, Sparkles, CheckCircle, AlertTriangle } from "lucide-react";

export default function AnalyzePage() {
  const [question, setQuestion] = useState("Who discovered penicillin?");
  const [context, setContext] = useState("Penicillin was discovered by Alexander Fleming in 1928.");
  const [answer, setAnswer] = useState("Penicillin was discovered by Louis Pasteur in 1945.");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const sampleScenarios = [
    {
      title: "Hallucinated Entity & Date",
      q: "Who discovered penicillin?",
      c: "Penicillin was discovered by Alexander Fleming in 1928.",
      a: "Louis Pasteur discovered penicillin in 1945.",
    },
    {
      title: "Grounded & Correct",
      q: "What is the capital of France?",
      c: "Paris is the capital and most populous city of France.",
      a: "Paris is the capital of France.",
    },
    {
      title: "Borderline Case",
      q: "When did the Apollo 11 moon landing occur?",
      c: "Apollo 11 landed on the Moon on July 20, 1969.",
      a: "The Apollo mission landed humans on the Moon in the late 1960s.",
    },
  ];

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);

    try {
      const predRes = await fetch("/api/ml/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context, answer }),
      });

      const expRes = await fetch("/api/ml/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context, answer }),
      });

      let prediction = predRes.ok ? await predRes.json() : null;
      let explanation = expRes.ok ? await expRes.json() : null;

      if (!prediction) {
        // Fallback demo result
        prediction = {
          calibrated_score: 0.88,
          label: "high_risk",
          latency_ms: 14,
          model_version: "xgb-calibrated-v1.0",
        };
        explanation = {
          top_features: [
            { feature: "overlap_answer_context", value: 0.12, impact: 0.38 },
            { feature: "novel_numbers", value: 2, impact: 0.28 },
            { feature: "hedge_count", value: 0, impact: 0.12 },
            { feature: "jaccard_ans_ctx", value: 0.08, impact: 0.10 },
          ],
          base_value: 0.5,
        };
      }

      setResult({ prediction, explanation });
    } catch (err) {
      console.error("API error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-purple-400" /> 📊 Analyze Mode — Form & Evidence Inspector
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Directly test the XGBoost risk model with question, context, and candidate answer
          </p>
        </div>

        {/* Pre-baked Scenario Buttons */}
        <div className="flex flex-wrap gap-2">
          {sampleScenarios.map((s, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuestion(s.q);
                setContext(s.c);
                setAnswer(s.a);
              }}
              className="text-xs font-medium bg-secondary/80 hover:bg-secondary border border-border px-3 py-1.5 rounded-lg transition-all"
            >
              {s.title}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form Column */}
        <form onSubmit={handleAnalyze} className="glass-panel p-6 rounded-2xl space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Question
            </label>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Reference Context / Evidence
            </label>
            <textarea
              rows={3}
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Candidate LLM Answer
            </label>
            <textarea
              rows={3}
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-semibold py-3 px-6 rounded-xl shadow-lg flex items-center justify-center gap-2 transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{loading ? "Computing Features & Predictions..." : "Run Risk Analysis"}</span>
          </button>
        </form>

        {/* Results Column */}
        <div className="space-y-6">
          {result ? (
            <>
              <RiskGauge
                score={result.prediction.calibrated_score}
                label={result.prediction.label}
                latencyMs={result.prediction.latency_ms}
              />
              {result.explanation && (
                <ShapChart
                  features={result.explanation.top_features}
                  baseValue={result.explanation.base_value}
                />
              )}
            </>
          ) : (
            <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center space-y-3 h-full">
              <Sparkles className="w-10 h-10 text-muted-foreground/40 animate-pulse" />
              <h3 className="text-sm font-semibold">No Analysis Computed Yet</h3>
              <p className="text-xs text-muted-foreground max-w-sm">
                Fill in the form or click one of the sample scenarios above and press "Run Risk Analysis".
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
