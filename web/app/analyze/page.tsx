"use client";

import React, { useState } from "react";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";
import { Play, Sparkles, AlertTriangle } from "lucide-react";

interface FeatureImpact {
  feature: string;
  value: number;
  impact: number;
}

interface Prediction {
  calibrated_score: number;
  label: string;
  latency_ms?: number;
  model_version?: string;
  risk_score?: number;
  thresholds?: Record<string, number>;
  warning?: string;
  features?: Record<string, number>;
}

interface Explanation {
  top_features?: FeatureImpact[];
  base_value?: number;
}

interface AnalysisResult {
  prediction: Prediction;
  explanation: Explanation | null;
}

export default function AnalyzePage() {
  const [question, setQuestion] = useState(
    "Are both The New Pornographers and Kings of Leon American rock bands?"
  );
  const [context, setContext] = useState(
    "The New Pornographers is a Canadian indie rock band; Kings of Leon is an American rock band."
  );
  const [answer, setAnswer] = useState(
    "Yes, both The New Pornographers and Kings of Leon are now American rock bands."
  );

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sampleScenarios = [
    {
      title: "Hallucinated Date",
      q: "By how many days per decade has the melt season lengthened in the shallowest of the world's major oceans?",
      c: "It has been established that the region is at its warmest for at least 40,000 years and the Arctic-wide melt season has lengthened at a rate of 5 days per decade (from 1979 to 2013), dominated by a later autumn freezeup. The Arctic Ocean is the smallest and shallowest of the world's five major oceans.",
      a: "The Arctic Ocean melt season has lengthened by 10 days per decade.",
    },
    {
      title: "Grounded & Correct",
      q: "Are both The New Pornographers and Kings of Leon American rock bands?",
      c: "The New Pornographers is a Canadian indie rock band; Kings of Leon is an American rock band.",
      a: "Yes, both The New Pornographers and Kings of Leon are now American rock bands.",
    },
    {
      title: "Borderline / Ambiguous",
      q: "This catlike alien space pirate is arch enemies to Space Ghost. What is his name?",
      c: "Featuring songs and skits by Space Ghost and his arch enemies Zorak and Brak. Brak is a fictional character and supervillain on the 1966 Hanna-Barbera animated series \"Space Ghost\", portrayed as a catlike alien space pirate trying to conquer the galaxy.",
      a: "Brak",
    },
  ];

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

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

      if (!predRes.ok) {
        const detail = (await predRes.json().catch(() => null))?.detail;
        setError(
          detail || `ML backend error (${predRes.status}). Is uvicorn running on port 8000?`
        );
        return;
      }

      const prediction = await predRes.json();
      const explanation = expRes.ok ? await expRes.json() : null;
      setResult({ prediction, explanation });
    } catch (err) {
      console.error("API error:", err);
      setError("Failed to reach the ML backend. Start it with: python -m uvicorn src.api.main:app --port 8000");
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
          {error && (
            <div className="glass-panel p-6 rounded-2xl border-l-4 border-l-red-500">
              <h3 className="text-sm font-semibold text-red-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> Analysis failed
              </h3>
              <p className="text-xs text-muted-foreground mt-2">{error}</p>
            </div>
          )}
          {result ? (
            <>
              <RiskGauge
                score={result.prediction.calibrated_score}
                label={result.prediction.label}
                latencyMs={result.prediction.latency_ms}
              />
              {result.explanation?.top_features && (
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
                Fill in the form or click one of the sample scenarios above and press &quot;Run Risk Analysis&quot;.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
