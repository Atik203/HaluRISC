import Link from "next/link";
import {
  AlertTriangle,
  ArrowRight,
  BarChart2,
  CircleCheck,
  ClipboardList,
  FlaskConical,
  MessageSquare,
  Play,
  Presentation,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { loadDashboardData, fmt, featureCount } from "@/lib/results";
import { DemoReadiness } from "@/components/demo-readiness";

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

export default function Home() {
  const d = loadDashboardData();
  const xgb = d.b2.comparison?.find((r) => r.model === "xgboost");
  const qaTest = d.b3.datasetMetrics?.ragtruth_qa_test;
  const target = d.b4.target;
  const sourceRef = target?.methods.raw_reference?.ece_mean;
  const targetEce = target?.methods.platt?.ece_mean;
  const judgeCost = d.legacy.judge?.cost_per_1000_usd;
  const haluriscCost = d.legacy.latency?.cost_per_1000_predictions_usd?.halurisc_local;
  const costRatio = judgeCost && haluriscCost ? Math.round(judgeCost / haluriscCost) : null;
  const featureCountValue = featureCount();
  const nFeatures = featureCountValue ?? 26;

  const review = d.b5.reviewCases ?? [];
  const hallucinated = review.find((c) => c.calibrated_score >= 0.7) ?? null;
  const grounded = review.find((c) => c.calibrated_score < 0.3) ?? null;

  const stats = [
    {
      value: xgb ? fmt(xgb.f1_mean) : "—",
      label: "In-domain F1",
      sub: "B2 · leakage-free grouped splits, 3 seeds",
    },
    {
      value: qaTest ? fmt(qaTest.f1_mean) : "—",
      label: "Zero-shot F1 on RAGTruth QA",
      sub: "B3 · external data, no retraining",
    },
    {
      value: sourceRef != null && targetEce != null ? `${fmt(sourceRef, 2)} → ${fmt(targetEce, 2)}` : "—",
      label: "ECE: source → target calibrated",
      sub: "B4 · calibration does not transfer for free",
    },
    {
      value: costRatio ? `${costRatio}×` : "—",
      label: "Cheaper than an LLM judge",
      sub: "measured cost per 1,000 predictions",
    },
  ];

  const steps = [
    {
      icon: ClipboardList,
      title: "Give it three things",
      body: "A question, the evidence you trust (pasted, uploaded, or searched), and the candidate answer. No model weights or activations needed.",
    },
    {
      icon: ScanSearch,
      title: "It measures grounding",
      body: `${nFeatures} features across seven groups: lexical overlap, entity coverage, numeric consistency, NLI contradiction, hedging, semantic drift, and length. A calibrated XGBoost turns them into a risk score.`,
    },
    {
      icon: ShieldCheck,
      title: "You get a verdict with citations",
      body: "Claim-level supported / contradicted / unsupported verdicts, SHAP attribution for every score, and the failure cases kept in the open.",
    },
  ];

  const destinations = [
    {
      href: "/analyze",
      icon: BarChart2,
      title: "Analyze mode",
      body: "Type or paste a case and inspect the score, thresholds, and SHAP values side by side.",
    },
    {
      href: "/chat",
      icon: MessageSquare,
      title: "Chat mode",
      body: "Converse with a grounded assistant; every answer is checked after it streams.",
    },
    {
      href: "/dashboard/overview",
      icon: FlaskConical,
      title: "Experiment dashboard",
      body: "All B2 to B5 tables, charts, and statistics, rendered straight from the frozen artifacts.",
    },
    {
      href: "/demo",
      icon: Presentation,
      title: "Presenter demo",
      body: "A fully offline walkthrough: no API key, no network, works on a clean clone.",
    },
  ];

  return (
    <div className="space-y-12 md:space-y-16 max-w-6xl mx-auto pb-4">
      {/* Hero */}
      <section className="grid lg:grid-cols-[1.15fr_0.85fr] gap-8 lg:gap-12 items-center pt-4 md:pt-10">
        <div className="space-y-6">
          <p className="eyebrow animate-rise flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
            Calibrated · Explainable · Lightweight
          </p>
          <h1 className="display-title text-4xl sm:text-5xl lg:text-6xl leading-[1.04] text-balance animate-rise [animation-delay:60ms]">
            Know when an answer stops being <span className="gradient-text">grounded</span>.
          </h1>
          <p className="text-base md:text-lg text-muted-foreground max-w-xl leading-relaxed animate-rise [animation-delay:120ms]">
            HaluRISC scores candidate answers against the evidence you provide. It runs on CPU in milliseconds, needs
            no model internals, and shows where its own limits are.
          </p>
          <div className="flex flex-wrap items-center gap-3 animate-rise [animation-delay:180ms]">
            <Link
              href="/analyze"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition-all hover:from-violet-500 hover:to-indigo-500"
            >
              <Play className="w-4 h-4 fill-current" aria-hidden /> Run an analysis
            </Link>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-secondary"
            >
              <MessageSquare className="w-4 h-4" aria-hidden /> Ask in chat
            </Link>
            <Link
              href="/demo"
              className="inline-flex items-center gap-1.5 px-2 py-2.5 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
            >
              Offline walkthrough <ArrowRight className="w-4 h-4" aria-hidden />
            </Link>
          </div>
          <ul className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground animate-rise [animation-delay:240ms]">
            {["CPU only, no GPU required", "Artifact-backed numbers, never staged", "Open failure cases"].map((t) => (
              <li key={t} className="inline-flex items-center gap-1.5">
                <CircleCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" aria-hidden /> {t}
              </li>
            ))}
          </ul>
        </div>

        {/* Real cases from the frozen review export — not staged */}
        <div className="space-y-4 animate-rise [animation-delay:200ms]">
          {hallucinated && (
            <article className="glass-panel rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2 text-xs font-semibold text-rose-600 dark:text-rose-400">
                  <AlertTriangle className="w-4 h-4" aria-hidden /> Contradicted evidence
                </span>
                <span className="rounded-full border border-rose-500/40 bg-rose-500/10 px-3 py-0.5 font-mono text-xs text-rose-600 dark:text-rose-400 tnum">
                  {Math.round(hallucinated.calibrated_score * 100)}% risk
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                <span className="font-semibold text-foreground">A: </span>
                {truncate(hallucinated.answer, 150)}
              </p>
              <p className="text-[11px] font-mono text-muted-foreground/80 truncate">
                top signal: {hallucinated.top5_shap_features.split(";")[0]?.trim()}
              </p>
            </article>
          )}
          {grounded && (
            <article className="glass-panel rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                  <ShieldCheck className="w-4 h-4" aria-hidden /> Grounded in evidence
                </span>
                <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-0.5 font-mono text-xs text-emerald-600 dark:text-emerald-400 tnum">
                  {Math.round(grounded.calibrated_score * 100)}% risk
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                <span className="font-semibold text-foreground">A: </span>
                {truncate(grounded.answer, 150)}
              </p>
              <p className="text-[11px] font-mono text-muted-foreground/80 truncate">
                top signal: {grounded.top5_shap_features.split(";")[0]?.trim()}
              </p>
            </article>
          )}
          <p className="text-[11px] text-muted-foreground text-center">
            Real cases from the B5 reviewer export · scores are the evidence-calibrated risk
          </p>
        </div>
      </section>

      {/* Evidence strip */}
      <section aria-label="Key results" className="glass-panel rounded-2xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 lg:divide-x divide-border/60 animate-fade">
        {stats.map((s) => (
          <div key={s.label} className="p-5 md:p-6 space-y-1.5">
            <div className="text-2xl md:text-3xl font-extrabold font-mono tnum tracking-tight">{s.value}</div>
            <div className="text-sm font-semibold">{s.label}</div>
            <div className="text-[11px] text-muted-foreground">{s.sub}</div>
          </div>
        ))}
      </section>

      {/* Booth preflight */}
      <DemoReadiness
        artifactsReady={Boolean(d.manifest)}
        modelsReady={featureCountValue != null}
        resultsReady={Boolean(d.b2.comparison && d.b4.metrics && d.b5.importance)}
      />

      {/* How it works */}
      <section className="space-y-6">
        <div className="flex items-end justify-between gap-4">
          <h2 className="display-title text-2xl md:text-3xl">How it works</h2>
          <Link href="/about" className="text-xs font-semibold text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
            Read the method <ArrowRight className="w-3.5 h-3.5" aria-hidden />
          </Link>
        </div>
        <div className="grid md:grid-cols-3 gap-4">
          {steps.map((s, i) => {
            const Icon = s.icon;
            return (
              <article key={s.title} className="glass-panel rounded-2xl p-5 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl surface-inset flex items-center justify-center">
                    <Icon className="w-4.5 h-4.5 text-violet-600 dark:text-purple-400" aria-hidden />
                  </div>
                  <span className="font-mono text-xs text-muted-foreground tnum">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <h3 className="text-sm font-bold">{s.title}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{s.body}</p>
              </article>
            );
          })}
        </div>
      </section>

      {/* Destinations */}
      <section className="space-y-6">
        <h2 className="display-title text-2xl md:text-3xl">Where to go next</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {destinations.map((dst) => {
            const Icon = dst.icon;
            return (
              <Link
                key={dst.href}
                href={dst.href}
                className="group glass-panel rounded-2xl p-5 flex items-start gap-4 transition-all hover:-translate-y-0.5 hover:border-primary/40"
              >
                <div className="w-10 h-10 shrink-0 rounded-xl surface-inset flex items-center justify-center">
                  <Icon className="w-5 h-5 text-violet-600 dark:text-purple-400" aria-hidden />
                </div>
                <div className="space-y-1">
                  <h3 className="text-sm font-bold flex items-center gap-2">
                    {dst.title}
                    <ArrowRight className="w-3.5 h-3.5 opacity-0 -translate-x-1 transition-all group-hover:opacity-100 group-hover:translate-x-0" aria-hidden />
                  </h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{dst.body}</p>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Honesty panel */}
      <section className="glass-panel rounded-2xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center gap-6 justify-between">
        <div className="space-y-2 max-w-2xl">
          <h2 className="text-base font-bold flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-violet-600 dark:text-purple-400" aria-hidden />
            It does not pretend to be perfect
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Zero-shot transfer to RAGTruth QA drops well below in-domain performance, and source calibration does not
            transfer to new data without refitting. Both effects ship in the dashboard with the exact numbers.
          </p>
        </div>
        <Link
          href="/dashboard/robustness"
          className="shrink-0 inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-secondary"
        >
          See the limits <ArrowRight className="w-4 h-4" aria-hidden />
        </Link>
      </section>
    </div>
  );
}
