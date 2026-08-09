"use client";

import React, { useCallback, useEffect, useState } from "react";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";
import { Play, Sparkles, AlertTriangle, ChevronDown, GitBranch, Cpu } from "lucide-react";

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
  feature_version?: string;
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

interface Meta {
  model_version?: string;
  feature_version?: string;
  thresholds?: Record<string, number>;
  warning?: string;
  device?: string;
  feature_groups?: Record<string, string[]>;
}

const LABEL_TEXT: Record<string, string> = {
  low_risk: "Low risk",
  medium_risk: "Medium risk",
  high_risk: "High risk",
};

const GROUP_LABELS: Record<string, string> = {
  length: "Length & style",
  lexical: "Lexical overlap",
  entity: "Entity coverage",
  nli: "NLI consistency",
  numeric: "Numeric consistency",
  hedging: "Hedging",
  semantic: "Semantic drift",
};

function riskTone(label: string) {
  if (label === "high_risk") return "text-rose-600 dark:text-rose-400 border-rose-500/40 bg-rose-500/10";
  if (label === "medium_risk") return "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/10";
  return "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/10";
}

function RiskPanel({
  result,
  meta,
  title,
}: {
  result: AnalysisResult;
  meta: Meta | null;
  title: string;
}) {
  const { prediction, explanation } = result;
  const thresholds = prediction.thresholds ?? meta?.thresholds;
  const features = prediction.features ?? {};
  const groups = meta?.feature_groups ?? {};
  const grouped: Array<[string, Array<[string, number]>]> = Object.entries(groups)
    .map(([group, cols]) => {
      const rows = cols
        .filter((c) => c in features)
        .map((c) => [c, features[c]] as [string, number]);
      return [group, rows] as [string, Array<[string, number]>];
    })
    .filter(([, rows]) => rows.length > 0);

  return (
    <section className="glass-panel p-5 rounded-2xl space-y-4">
      <h3 className="text-sm font-bold flex items-center gap-2 text-muted-foreground">
        <GitBranch className="w-4 h-4 text-violet-600 dark:text-purple-400" aria-hidden /> {title}
      </h3>

      <RiskGauge score={prediction.calibrated_score} label={prediction.label} latencyMs={prediction.latency_ms} />

      {thresholds && (
        <div className="flex flex-wrap gap-2 justify-center text-[10px] font-mono">
          <span className="px-2 py-0.5 rounded-full border border-border bg-secondary/40">low &lt; {thresholds.low?.toFixed(2)}</span>
          <span className="px-2 py-0.5 rounded-full border border-border bg-secondary/40">
            medium {thresholds.low?.toFixed(2)}–{thresholds.medium?.toFixed(2)}
          </span>
          <span className="px-2 py-0.5 rounded-full border border-border bg-secondary/40">high ≥ {thresholds.medium?.toFixed(2)}</span>
        </div>
      )}

      <p className="text-xs text-center">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-semibold ${riskTone(prediction.label)}`}>
          {LABEL_TEXT[prediction.label] ?? prediction.label}
        </span>
      </p>

      {prediction.warning && (
        <p className="text-[11px] text-amber-600/90 dark:text-amber-400/90 bg-amber-500/5 border border-amber-500/20 rounded-xl px-3 py-2">
          ⚠ {prediction.warning}
        </p>
      )}

      <p className="text-[10px] font-mono text-muted-foreground text-center">
        model {prediction.model_version ?? meta?.model_version ?? "—"} · features {prediction.feature_version ?? meta?.feature_version ?? "—"} ·{" "}
        {prediction.latency_ms != null && `latency ${prediction.latency_ms.toFixed(0)} ms · `}
        <Cpu className="inline w-3 h-3" aria-hidden /> {meta?.device ?? "cpu"}
      </p>

      {explanation?.top_features && (
        <div className="space-y-1">
          <ShapChart features={explanation.top_features} baseValue={explanation.base_value} />
          <p className="text-[10px] text-muted-foreground">
            SHAP values reflect the <strong>raw XGBoost model</strong> (feature attribution), not the calibrated score.
          </p>
        </div>
      )}

      {grouped.length > 0 && (
        <details className="group border border-border/60 rounded-xl bg-secondary/20">
          <summary className="flex items-center justify-between px-4 py-2.5 cursor-pointer text-xs font-semibold text-muted-foreground hover:text-foreground">
            <span>Expandable features — {grouped.reduce((n, [, rows]) => n + rows.length, 0)} of {Object.keys(features).length}</span>
            <ChevronDown className="w-4 h-4 transition-transform group-open:rotate-180" aria-hidden />
          </summary>
          <div className="px-4 pb-3 space-y-3">
            {grouped.map(([group, rows]) => (
              <div key={group}>
                <h4 className="text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-purple-400 mb-1">
                  {GROUP_LABELS[group] ?? group}
                </h4>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-0.5">
                  {rows.map(([name, value]) => (
                    <li key={name} className="flex justify-between text-[11px] gap-3">
                      <span className="text-muted-foreground truncate">{name}</span>
                      <span className="font-mono">{Number(value).toFixed(4)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

export default function AnalyzePage() {
  const [question, setQuestion] = useState(
    "Are both The New Pornographers and Kings of Leon American rock bands?"
  );
  const [context, setContext] = useState(
    "The New Pornographers is a Canadian indie rock band; Kings of Leon is an American rock band."
  );
  const [answerA, setAnswerA] = useState(
    "Yes, both The New Pornographers and Kings of Leon are now American rock bands."
  );
  const [answerB, setAnswerB] = useState("");
  const [compare, setCompare] = useState(false);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{ a: AnalysisResult; b?: AnalysisResult } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    fetch("/api/ml/meta", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => setMeta(m ?? null))
      .catch(() => setMeta(null));
  }, []);

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

  const analyzeOne = useCallback(async (answer: string): Promise<AnalysisResult> => {
    const payload = { question, context, answer };
    const [predRes, expRes] = await Promise.all([
      fetch("/api/ml/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch("/api/ml/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ]);
    if (!predRes.ok) {
      const detail = (await predRes.json().catch(() => null))?.detail;
      throw new Error(detail || `ML backend error (${predRes.status}). Is uvicorn running on port 8000?`);
    }
    const prediction = (await predRes.json()) as Prediction;
    const explanation = expRes.ok ? ((await expRes.json()) as Explanation) : null;
    return { prediction, explanation };
  }, [question, context]);

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (compare && !answerB.trim()) {
      setError("Compare mode needs a second answer (Answer B).");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const a = await analyzeOne(answerA);
      const b = compare ? await analyzeOne(answerB) : undefined;
      setResults({ a, b });
    } catch (err) {
      console.error("API error:", err);
      setError(err instanceof Error ? err.message : "Failed to reach the ML backend.");
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
            <Sparkles className="w-6 h-6 text-violet-600 dark:text-purple-400" /> 📊 Analyze Mode — Form &amp; Evidence Inspector
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Directly test the calibrated XGBoost risk model — including two-answer comparison
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {sampleScenarios.map((s, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setQuestion(s.q);
                setContext(s.c);
                setAnswerA(s.a);
                setAnswerB("");
                setCompare(false);
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
            <label htmlFor="analyze-question" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Question
            </label>
            <input
              id="analyze-question"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <div>
            <label htmlFor="analyze-context" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Reference Context / Evidence
            </label>
            <textarea
              id="analyze-context"
              rows={3}
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <div>
            <label htmlFor="analyze-answer-a" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              Candidate Answer A
            </label>
            <textarea
              id="analyze-answer-a"
              rows={3}
              value={answerA}
              onChange={(e) => setAnswerA(e.target.value)}
              className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>

          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => {
                setCompare(e.target.checked);
                if (!e.target.checked) setAnswerB("");
              }}
              className="accent-violet-600"
            />
            Compare two answers (Answer B)
          </label>

          {compare && (
            <div>
              <label htmlFor="analyze-answer-b" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Candidate Answer B
              </label>
              <textarea
                id="analyze-answer-b"
                rows={3}
                value={answerB}
                onChange={(e) => setAnswerB(e.target.value)}
                className="w-full bg-secondary/50 border border-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                required
              />
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-semibold py-3 px-6 rounded-xl shadow-lg flex items-center justify-center gap-2 transition-all disabled:opacity-60"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{loading ? "Computing Features & Predictions..." : compare ? "Run Comparison" : "Run Risk Analysis"}</span>
          </button>
        </form>

        {/* Results Column */}
        <div className="space-y-6" aria-live="polite" aria-busy={loading}>
          {error && (
            <div className="glass-panel p-6 rounded-2xl border-l-4 border-l-red-500" role="alert">
              <h3 className="text-sm font-semibold text-red-600 dark:text-red-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> Analysis failed
              </h3>
              <p className="text-xs text-muted-foreground mt-2">{error}</p>
            </div>
          )}
          {loading && (
            <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center space-y-3">
              <Sparkles className="w-10 h-10 text-muted-foreground/40 animate-pulse" />
              <h3 className="text-sm font-semibold">Computing features (NLI + embeddings)…</h3>
            </div>
          )}
          {results && (
            <div className={`grid gap-6 ${results.b ? "lg:grid-cols-2" : "grid-cols-1"}`}>
              <RiskPanel result={results.a} meta={meta} title="Answer A" />
              {results.b && <RiskPanel result={results.b} meta={meta} title="Answer B" />}
            </div>
          )}
          {!loading && !results && !error && (
            <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center space-y-3 h-full">
              <Sparkles className="w-10 h-10 text-muted-foreground/40 animate-pulse" />
              <h3 className="text-sm font-semibold">No Analysis Computed Yet</h3>
              <p className="text-xs text-muted-foreground max-w-sm">
                Fill in the form or click one of the sample scenarios above and press &quot;Run Risk Analysis&quot;. Tick
                &quot;Compare two answers&quot; to score Answer B against the same question and evidence.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
