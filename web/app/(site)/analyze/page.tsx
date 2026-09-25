"use client";

import React, { useCallback, useEffect, useState } from "react";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapChart } from "@/components/shap-chart";
import { ModelComparisonCard, type ModelComparisonInput } from "@/components/model-comparison-card";
import { Switch } from "@/components/ui/switch";
import { MlStatus } from "@/components/ml-status";
import {
  AlertTriangle,
  ChevronDown,
  CircleCheck,
  ClipboardCopy,
  Cpu,
  GitBranch,
  Hash,
  Info,
  ListChecks,
  Play,
  RefreshCw,
  Scale,
  Sparkles,
} from "lucide-react";
import { toast } from "@/components/ui/toast";

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
  thresholds?: { low?: number; medium?: number };
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
  thresholds?: { low?: number; medium?: number };
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
  if (label === "high_risk") return "text-rose-700 dark:text-rose-300 border-rose-500/40 bg-rose-500/10";
  if (label === "medium_risk") return "text-amber-700 dark:text-amber-300 border-amber-500/40 bg-amber-500/10";
  return "text-emerald-700 dark:text-emerald-300 border-emerald-500/40 bg-emerald-500/10";
}

function RiskPanel({ result, meta, title }: { result: AnalysisResult; meta: Meta | null; title: string }) {
  const { prediction, explanation } = result;
  const thresholds = prediction.thresholds ?? meta?.thresholds;
  const features = prediction.features ?? {};
  const groups = meta?.feature_groups ?? {};
  const grouped: Array<[string, Array<[string, number]>]> = Object.entries(groups)
    .map(([group, cols]) => {
      const rows = cols.filter((c) => c in features).map((c) => [c, features[c]] as [string, number]);
      return [group, rows] as [string, Array<[string, number]>];
    })
    .filter(([, rows]) => rows.length > 0);

  return (
    <section className="glass-panel rounded-2xl p-5 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-bold">
          <GitBranch className="w-4 h-4 text-violet-600 dark:text-purple-400" aria-hidden /> {title}
        </h3>
        <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-semibold ${riskTone(prediction.label)}`}>
          {LABEL_TEXT[prediction.label] ?? prediction.label}
        </span>
      </div>

      <RiskGauge score={prediction.calibrated_score} label={prediction.label} thresholds={thresholds} />

      {thresholds && (
        <div className="flex flex-wrap justify-center gap-2 text-[10px] font-mono text-muted-foreground">
          <span className="rounded-full border border-border bg-secondary/40 px-2 py-0.5">low &lt; {thresholds.low?.toFixed(2)}</span>
          <span className="rounded-full border border-border bg-secondary/40 px-2 py-0.5">
            medium {thresholds.low?.toFixed(2)}–{thresholds.medium?.toFixed(2)}
          </span>
          <span className="rounded-full border border-border bg-secondary/40 px-2 py-0.5">high ≥ {thresholds.medium?.toFixed(2)}</span>
        </div>
      )}

      <details className="group">
        <summary className="flex cursor-pointer items-center justify-center gap-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground">
          <Info className="h-3.5 w-3.5" aria-hidden /> How to read this result
        </summary>
        <ul className="mt-2 space-y-1.5 text-[11px] leading-relaxed text-muted-foreground">
          <li>
            The percentage is a calibrated probability. On the calibration data, answers that scored near this value
            were hallucinated about this often.
          </li>
          <li>
            The notches on the gauge are the low, medium, and high decision cutoffs. The needle points at this
            answer&apos;s score, and the label above uses the same cutoffs.
          </li>
          <li>
            Low risk is not a guarantee of correctness. It means the measured evidence did not look contradictory or
            unsupported.
          </li>
          <li>
            The SHAP chart under the gauge explains what moved the raw score. It is separate from the calibrated
            percentage, so the two can point in different directions.
          </li>
        </ul>
      </details>

      {prediction.warning && (
        <p className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
          {prediction.warning}
        </p>
      )}

      {explanation?.top_features && (
        <div className="space-y-1.5">
          <ShapChart features={explanation.top_features} baseValue={explanation.base_value} />
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            SHAP values explain the <strong className="font-semibold text-foreground">raw XGBoost model</strong>, not the
            calibrated score.
          </p>
        </div>
      )}

      {grouped.length > 0 && (
        <details className="group rounded-xl border border-border/60 bg-secondary/20">
          <summary className="flex cursor-pointer items-center justify-between px-4 py-2.5 text-xs font-semibold text-muted-foreground hover:text-foreground">
            <span>
              All features used — {grouped.reduce((n, [, rows]) => n + rows.length, 0)} of {Object.keys(features).length}
            </span>
            <ChevronDown className="w-4 h-4 transition-transform group-open:rotate-180" aria-hidden />
          </summary>
          <div className="space-y-3 px-4 pb-3">
            {grouped.map(([group, rows]) => (
              <div key={group}>
                <h4 className="eyebrow mb-1 text-violet-700 dark:text-purple-300">{GROUP_LABELS[group] ?? group}</h4>
                <ul className="grid grid-cols-1 gap-x-6 gap-y-0.5 sm:grid-cols-2">
                  {rows.map(([name, value]) => (
                    <li key={name} className="flex justify-between gap-3 text-[11px]">
                      <span className="truncate text-muted-foreground">{name}</span>
                      <span className="font-mono tnum">{Number(value).toFixed(4)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </details>
      )}

      <p className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[10px] font-mono text-muted-foreground">
        <span>
          model {prediction.model_version ?? meta?.model_version ?? "—"} · features{" "}
          {prediction.feature_version ?? meta?.feature_version ?? "—"}
        </span>
        {prediction.latency_ms != null && <span>latency {prediction.latency_ms.toFixed(0)} ms</span>}
        <span className="inline-flex items-center gap-1">
          <Cpu className="inline w-3 h-3" aria-hidden /> {meta?.device ?? "cpu"}
        </span>
      </p>
    </section>
  );
}

const SAMPLE_SCENARIOS = [
  {
    title: "Hallucinated number",
    hint: "Context says 5 days, answer says 10",
    icon: Hash,
    q: "By how many days per decade has the melt season lengthened in the shallowest of the world's major oceans?",
    c: "It has been established that the region is at its warmest for at least 40,000 years and the Arctic-wide melt season has lengthened at a rate of 5 days per decade (from 1979 to 2013), dominated by a later autumn freezeup. The Arctic Ocean is the smallest and shallowest of the world's five major oceans.",
    a: "The Arctic Ocean melt season has lengthened by 10 days per decade.",
  },
  {
    title: "Grounded answer",
    hint: "Fully supported by the context",
    icon: CircleCheck,
    q: "Which president was present for the Mexican Civil War who originated from Oaxaca?",
    c: "Agnes Salm-Salm was the American wife of Prince Felix zu Salm-Salm. Benito Pablo Juarez Garcia was a Mexican lawyer and liberal politician of Zapotec origin from Oaxaca.",
    a: "Benito Pablo Juarez Garcia",
  },
  {
    title: "Borderline case",
    hint: "Close to the decision boundary",
    icon: Scale,
    q: "This catlike alien space pirate is arch enemies to Space Ghost. What is his name?",
    c: "Featuring songs and skits by Space Ghost and his arch enemies Zorak and Brak. Brak is a fictional character and supervillain on the 1966 Hanna-Barbera animated series \"Space Ghost\", portrayed as a catlike alien space pirate trying to conquer the galaxy.",
    a: "Brak",
  },
];

export default function AnalyzePage() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [answerA, setAnswerA] = useState("");
  const [answerB, setAnswerB] = useState("");
  const [compare, setCompare] = useState(false);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{ a: AnalysisResult; b?: AnalysisResult } | null>(null);
  const [submitted, setSubmitted] = useState<{ a: ModelComparisonInput; b?: ModelComparisonInput } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    fetch("/api/ml/meta", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => setMeta(m ?? null))
      .catch(() => setMeta(null));
  }, []);

  const analyzeInput = useCallback(
    async (input: { question: string; context: string; answer: string }): Promise<AnalysisResult> => {
      const [predRes, expRes] = await Promise.all([
        fetch("/api/ml/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
        }),
        fetch("/api/ml/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
        }),
      ]);
      if (!predRes.ok) {
        const detail = (await predRes.json().catch(() => null))?.detail;
        throw new Error(detail || `ML backend error (${predRes.status}).`);
      }
      const prediction = (await predRes.json()) as Prediction;
      const explanation = expRes.ok ? ((await expRes.json()) as Explanation) : null;
      return { prediction, explanation };
    },
    [],
  );

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (compare && !answerB.trim()) {
      setError("Compare mode needs a second answer (Answer B).");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    const inputA: ModelComparisonInput = { question, context, answer: answerA };
    const inputB: ModelComparisonInput | undefined = compare ? { question, context, answer: answerB } : undefined;
    try {
      const a = await analyzeInput(inputA);
      const b = inputB ? await analyzeInput(inputB) : undefined;
      setResults({ a, b });
      setSubmitted({ a: inputA, b: inputB });
    } catch (err) {
      console.error("API error:", err);
      setError(err instanceof Error ? err.message : "Failed to reach the ML backend.");
    } finally {
      setLoading(false);
    }
  };

  interface SummaryRow {
    title: string;
    label: string;
    score: number;
    latency?: number;
  }
  const [summary, setSummary] = useState<SummaryRow[] | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  const runAllExamples = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    setSummary(null);
    try {
      const rows: SummaryRow[] = [];
      let firstResult: AnalysisResult | null = null;
      for (let i = 0; i < SAMPLE_SCENARIOS.length; i++) {
        const scenario = SAMPLE_SCENARIOS[i];
        setProgress(`Running example ${i + 1} of ${SAMPLE_SCENARIOS.length}: ${scenario.title}`);
        const result = await analyzeInput({ question: scenario.q, context: scenario.c, answer: scenario.a });
        if (!firstResult) firstResult = result;
        rows.push({
          title: scenario.title,
          label: result.prediction.label,
          score: result.prediction.calibrated_score,
          latency: result.prediction.latency_ms,
        });
      }
      setSummary(rows);
      applyScenario(SAMPLE_SCENARIOS[0]);
      if (firstResult) setResults({ a: firstResult });
      setSubmitted({
        a: {
          question: SAMPLE_SCENARIOS[0].q,
          context: SAMPLE_SCENARIOS[0].c,
          answer: SAMPLE_SCENARIOS[0].a,
        },
      });
      toast("Three examples scored. Copy the summary when you need it.");
    } catch (err) {
      console.error("API error:", err);
      setError(err instanceof Error ? err.message : "Failed to reach the ML backend.");
    } finally {
      setProgress(null);
      setLoading(false);
    }
  };

  const copySummary = async () => {
    if (!summary) return;
    const lines = summary.map(
      (row) =>
        `${row.title}: ${LABEL_TEXT[row.label] ?? row.label} · ${Math.round(row.score * 100)}% calibrated${
          row.latency != null ? ` · ${row.latency.toFixed(0)} ms` : ""
        }`,
    );
    try {
      await navigator.clipboard.writeText(`HaluRISC example sweep\n${lines.join("\n")}`);
      toast("Summary copied to clipboard");
    } catch {
      toast("Copy failed. The browser blocked clipboard access.", "error");
    }
  };

  const applyScenario = (s: (typeof SAMPLE_SCENARIOS)[number]) => {
    setQuestion(s.q);
    setContext(s.c);
    setAnswerA(s.a);
    setAnswerB("");
    setCompare(false);
  };

  const deltaPts =
    results?.b != null
      ? Math.round((results.b.prediction.calibrated_score - results.a.prediction.calibrated_score) * 100)
      : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel rounded-2xl p-5 md:p-6 flex flex-wrap items-center justify-between gap-4 animate-fade">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl surface-inset flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-violet-600 dark:text-purple-400" aria-hidden />
          </div>
          <div>
            <h1 className="display-title text-2xl">Analyze mode</h1>
            <p className="text-xs text-muted-foreground">
              Score a candidate answer against its evidence, with thresholds and SHAP attribution
            </p>
          </div>
        </div>
        <MlStatus />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <form onSubmit={handleAnalyze} className="glass-panel rounded-2xl p-5 md:p-6 space-y-5">
          <div className="space-y-2">
            <span className="eyebrow">Start from an example</span>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_SCENARIOS.map((s) => {
                const Icon = s.icon;
                return (
                  <button
                    key={s.title}
                    type="button"
                    onClick={() => applyScenario(s)}
                    title={s.hint}
                    className="inline-flex items-center gap-2 rounded-xl border border-border/70 bg-secondary/50 px-3 py-2 text-xs font-semibold transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:bg-secondary"
                  >
                    <Icon className="w-3.5 h-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
                    {s.title}
                  </button>
                );
              })}
              <button
                type="button"
                onClick={runAllExamples}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-xl border border-primary/40 bg-primary/10 px-3 py-2 text-xs font-semibold transition-colors hover:bg-primary/15 disabled:opacity-60"
              >
                <ListChecks className="w-3.5 h-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
                Run all three
              </button>
            </div>
            {progress && <p className="text-[11px] text-muted-foreground">{progress}</p>}
          </div>

          <div className="space-y-4">
            <div>
              <label htmlFor="analyze-question" className="eyebrow mb-1.5 block">
                Question
              </label>
              <input
                id="analyze-question"
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What is the user asking?"
                className="w-full rounded-xl border border-border bg-secondary/50 px-4 py-2.5 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
                required
              />
            </div>

            <div>
              <label htmlFor="analyze-context" className="eyebrow mb-1.5 block">
                Reference context / evidence
              </label>
              <textarea
                id="analyze-context"
                rows={4}
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Paste the trusted text the answer should be grounded in."
                className="w-full rounded-xl border border-border bg-secondary/50 px-4 py-2.5 text-sm leading-relaxed transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
                required
              />
            </div>

            <div>
              <label htmlFor="analyze-answer-a" className="eyebrow mb-1.5 block">
                Candidate answer A
              </label>
              <textarea
                id="analyze-answer-a"
                rows={3}
                value={answerA}
                onChange={(e) => setAnswerA(e.target.value)}
                placeholder="Paste the model answer to check."
                className="w-full rounded-xl border border-border bg-secondary/50 px-4 py-2.5 text-sm leading-relaxed transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
                required
              />
            </div>

            <Switch checked={compare} onChange={setCompare} label="Compare two answers" icon={<Scale className="w-3.5 h-3.5" aria-hidden />} />

            {compare && (
              <div className="animate-fade">
                <label htmlFor="analyze-answer-b" className="eyebrow mb-1.5 block">
                  Candidate answer B
                </label>
                <textarea
                  id="analyze-answer-b"
                  rows={3}
                  value={answerB}
                  onChange={(e) => setAnswerB(e.target.value)}
                  placeholder="Paste the second answer to score against the same question and evidence."
                  className="w-full rounded-xl border border-border bg-secondary/50 px-4 py-2.5 text-sm leading-relaxed transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-6 py-3 font-semibold text-white shadow-lg shadow-violet-600/20 transition-all hover:from-violet-500 hover:to-indigo-500 disabled:opacity-60"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" aria-hidden /> : <Play className="w-4 h-4 fill-current" aria-hidden />}
            <span>{loading ? "Computing features and predictions…" : compare ? "Run comparison" : "Run risk analysis"}</span>
          </button>
        </form>

        {/* Results */}
        <div className="space-y-6" aria-live="polite" aria-busy={loading}>
          {error && (
            <div className="glass-panel rounded-2xl border-l-4 border-l-rose-500 p-5 space-y-3" role="alert">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-rose-700 dark:text-rose-300">
                <AlertTriangle className="w-4 h-4" aria-hidden /> Analysis failed
              </h3>
              <p className="text-xs leading-relaxed text-muted-foreground">
                The local ML service did not answer. Make sure the FastAPI backend is running on port 8000, then try again.
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => handleAnalyze()}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-xs font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500"
                >
                  <RefreshCw className="w-3.5 h-3.5" aria-hidden /> Retry
                </button>
              </div>
              <details>
                <summary className="cursor-pointer text-[11px] font-semibold text-muted-foreground hover:text-foreground">
                  Technical details
                </summary>
                <p className="mt-2 rounded-lg surface-inset px-3 py-2 font-mono text-[11px]">{error}</p>
              </details>
            </div>
          )}

          {loading && (
            <div className="glass-panel rounded-2xl p-5 space-y-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-violet-500" aria-hidden />
                Extracting features (NLI + embeddings) and scoring…
              </div>
              <div className="space-y-3" aria-hidden>
                <div className="h-32 rounded-2xl bg-secondary/50 animate-pulse-soft" />
                <div className="h-3 w-40 rounded-full bg-secondary/60 animate-pulse-soft" />
                <div className="h-3 w-2/3 rounded-full bg-secondary/60 animate-pulse-soft [animation-delay:120ms]" />
                <div className="h-3 w-1/2 rounded-full bg-secondary/60 animate-pulse-soft [animation-delay:240ms]" />
              </div>
            </div>
          )}

          {summary && (
            <div className="glass-panel space-y-3 rounded-2xl p-4 animate-fade">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-bold">Example sweep</h3>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={copySummary}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-secondary/60 px-3 py-1.5 text-[11px] font-semibold transition-colors hover:bg-secondary"
                  >
                    <ClipboardCopy className="h-3.5 w-3.5" aria-hidden /> Copy summary
                  </button>
                  <button
                    type="button"
                    onClick={() => setSummary(null)}
                    className="rounded-xl px-2 py-1.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:text-foreground"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <ul className="divide-y divide-border/40">
                {summary.map((row) => (
                  <li key={row.title} className="flex flex-wrap items-center justify-between gap-2 py-2 text-xs">
                    <span className="font-semibold">{row.title}</span>
                    <span className="flex items-center gap-2">
                      <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${riskTone(row.label)}`}>
                        {LABEL_TEXT[row.label] ?? row.label}
                      </span>
                      <span className="font-mono tnum">{Math.round(row.score * 100)}%</span>
                      {row.latency != null && (
                        <span className="font-mono text-[10px] text-muted-foreground">{row.latency.toFixed(0)} ms</span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {results && (
            <div className="space-y-4">
              {results.b && deltaPts != null && (
                <div className="glass-panel rounded-2xl p-4 flex items-center gap-3 animate-fade">
                  <Scale className="w-5 h-5 shrink-0 text-violet-600 dark:text-purple-400" aria-hidden />
                  <p className="text-sm leading-relaxed">
                    {deltaPts === 0 ? (
                      <>Answer A and Answer B score <strong>the same risk</strong> on this evidence.</>
                    ) : (
                      <>
                        Answer <strong>{deltaPts > 0 ? "A" : "B"}</strong> is{" "}
                        <strong className="font-mono tnum">{Math.abs(deltaPts)} points</strong> safer than Answer{" "}
                        <strong>{deltaPts > 0 ? "B" : "A"}</strong>
                      </>
                    )}
                    <span className="text-muted-foreground">
                      {" "}
                      · {Math.round(results.a.prediction.calibrated_score * 100)}% vs{" "}
                      {Math.round(results.b.prediction.calibrated_score * 100)}% calibrated risk
                    </span>
                  </p>
                </div>
              )}
              <div className={`grid gap-6 ${results.b ? "lg:grid-cols-2" : "grid-cols-1"}`}>
                <RiskPanel result={results.a} meta={meta} title="Answer A" />
                {results.b && <RiskPanel result={results.b} meta={meta} title="Answer B" />}
              </div>

              {submitted && (
                <div className={`grid gap-6 ${submitted.b ? "lg:grid-cols-2" : "grid-cols-1"}`}>
                  <ModelComparisonCard
                    input={submitted.a}
                    title={submitted.b ? "Answer A · model comparison" : "Model comparison"}
                  />
                  {submitted.b && <ModelComparisonCard input={submitted.b} title="Answer B · model comparison" />}
                </div>
              )}
            </div>
          )}

          {!loading && !results && !error && (
            <div className="glass-panel flex h-full min-h-[22rem] flex-col items-center justify-center gap-4 rounded-2xl p-8 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl surface-inset">
                <Sparkles className="w-5 h-5 text-violet-600 dark:text-purple-400 animate-pulse-soft" aria-hidden />
              </div>
              <h3 className="display-title text-xl">No analysis yet</h3>
              <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">
                Fill in the question, the evidence, and a candidate answer. Or start from one of the three examples above
                and press <span className="font-semibold text-foreground">Run risk analysis</span>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
