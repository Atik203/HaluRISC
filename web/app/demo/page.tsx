import Link from "next/link";
import { ArrowRight, Presentation, ShieldCheck, AlertTriangle, Gauge, Scale } from "lucide-react";
import { loadDashboardData, fmt } from "@/lib/results";
import { Panel, Figure, DataTable } from "@/components/dashboard/panel";

/**
 * Presenter demo — fully offline: every number/figure is rendered from
 * artifacts/ at request time. No API calls, no OpenAI key, no network.
 */
export default function DemoPage() {
  const d = loadDashboardData();
  const review = d.b5.reviewCases ?? [];
  const errors = d.b3.errorCases ?? [];
  const b5Failures = d.b5.failureCases ?? [];
  const inDomain = d.b2.comparison?.find((r) => r.model === "xgboost");
  const qaTest = d.b3.datasetMetrics?.ragtruth_qa_test;
  const faith = d.b3.datasetMetrics?.faithbench;
  const target = d.b4.target;
  const sourceRef = target?.methods.raw_reference?.ece_mean;
  const targetEce = target?.methods.platt?.ece_mean;

  const grounded = review.find((c) => c.calibrated_score < 0.3);
  const hallucinated = review.find((c) => c.calibrated_score >= 0.7);
  const borderline = review.find(
    (c) => c.calibrated_score >= 0.35 && c.calibrated_score <= 0.65,
  );

  const failure = b5Failures[0] ?? null;
  const b3Error = errors.find((c) => c.source_dataset === "ragtruth") ?? errors[0] ?? null;

  const stats = [
    { label: "In-domain test F1 (B2, leakage-free)", value: inDomain ? fmt(inDomain.f1_mean) : "—" },
    { label: "RAGTruth QA zero-shot F1 (B3)", value: qaTest ? fmt(qaTest.f1_mean) : "—" },
    { label: "FaithBench zero-shot F1 (B3)", value: faith ? fmt(faith.f1_mean) : "—" },
    { label: "ECE on QA test: raw → target-calibrated (B4)", value: sourceRef != null && targetEce != null ? `${fmt(sourceRef)} → ${fmt(targetEce)}` : "—" },
  ];

  const riskTone = (s: number) =>
    s >= 0.7 ? "text-rose-600 dark:text-rose-400 border-rose-500/40 bg-rose-500/10" :
    s >= 0.3 ? "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/10" :
    "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/10";
  const riskWord = (s: number) => (s >= 0.7 ? "High risk" : s >= 0.3 ? "Medium risk" : "Low risk");

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="glass-panel p-6 md:p-8 rounded-2xl">
        <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
          <Presentation className="w-6 h-6 text-violet-600 dark:text-purple-400" /> Presenter Demo
        </h1>
        <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
          A self-contained walkthrough: every score, table, and figure below is rendered from the frozen{" "}
          <code className="font-mono">artifacts/</code> — no API calls, no OpenAI key, no network. Works on a clean
          clone after unzipping the Colab artifact bundle.
        </p>
      </div>

      <Panel title="The problem in one line">
        <p className="text-sm leading-relaxed text-foreground/90">
          Black-box LLMs hallucinate; you cannot inspect their weights. HaluRISC scores candidate answers against the
          provided context with <strong>26 calibrated, explainable features</strong> — and measures honestly where that
          works and where it breaks.
        </p>
        <div className="flex flex-wrap gap-2 pt-2">
          {stats.map((s) => (
            <div key={s.label} className="glass-panel px-4 py-3 rounded-xl flex-1 min-w-[220px]">
              <div className="text-lg font-extrabold font-mono">{s.value}</div>
              <div className="text-[11px] text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Three real cases (from the B5 reviewer export)" subtitle="Sampled from b5_review_cases.json — real scores, no staging data.">
        {[grounded, hallucinated, borderline].map((case_) =>
          case_ ? (
            <div key={case_.sample_id} className="border border-border/60 rounded-xl p-4 space-y-2 bg-secondary/20">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-bold flex items-center gap-2">
                  {case_.calibrated_score < 0.3 ? (
                    <ShieldCheck className="w-4 h-4 text-emerald-500" aria-hidden />
                  ) : case_.calibrated_score >= 0.7 ? (
                    <AlertTriangle className="w-4 h-4 text-rose-500" aria-hidden />
                  ) : (
                    <Scale className="w-4 h-4 text-amber-500" aria-hidden />
                  )}
                  {riskWord(case_.calibrated_score)}
                </h3>
                <span className={`px-3 py-0.5 rounded-full border font-mono text-xs ${riskTone(case_.calibrated_score)}`}>
                  calibrated {case_.calibrated_score.toFixed(3)} · raw {case_.raw_score.toFixed(3)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                <span className="font-semibold text-foreground">Q:</span> {case_.question.length > 160 ? `${case_.question.slice(0, 160)}…` : case_.question}
              </p>
              <p className="text-xs text-muted-foreground">
                <span className="font-semibold text-foreground">A:</span> {case_.answer.length > 220 ? `${case_.answer.slice(0, 220)}…` : case_.answer}
              </p>
              <p className="text-[11px] text-muted-foreground">
                Top-5 SHAP features: <code className="font-mono">{case_.top5_shap_features}</code> · label {case_.label}
              </p>
            </div>
          ) : null,
        )}
      </Panel>

      <Panel title="A real failure (B5 perturbation protocol)" subtitle="Controlled edit that flipped or moved the model's score.">
        {failure ? (
          <p className="text-xs text-muted-foreground leading-relaxed">
            Sample <code className="font-mono">{failure.sample_id}</code> under <strong>{failure.perturbation}</strong>{" "}
            perturbation: raw score {fmt(failure.raw_score)} → {fmt(failure.perturbed_score)} (Δ{" "}
            {failure.score_delta >= 0 ? "+" : ""}
            {fmt(failure.score_delta)}), SHAP top-1 {failure.top1_flip ? "flipped" : "stable"} (
            {failure.orig_top1} → {failure.pert_top1}).
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">b5_failure_cases.json not present.</p>
        )}
        {b3Error && (
          <p className="text-xs text-muted-foreground leading-relaxed pt-2 border-t border-border/40">
            Zero-shot error case ({b3Error.source_dataset}): score {fmt(b3Error.raw_score)}, true label {b3Error.label}.
            {b3Error.text_redacted ? " FaithBench text redacted (CC BY-NC-SA)." : ""}
          </p>
        )}
      </Panel>

      <Panel title="Calibration under shift — the key evidence (B4)">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Figure src="/api/figures/b4/calibration_shift.png" alt="Calibration shift across subsets and methods" className="w-full" />
          <Figure src="/api/figures/b4/reliability_diagrams.png" alt="Reliability diagrams for HaluEval, RAGTruth QA and FaithBench" className="w-full" />
        </div>
        <p className="text-xs text-muted-foreground pt-2">
          Source calibration does not transfer: ECE ≈ {sourceRef != null ? fmt(sourceRef) : "—"} on RAGTruth QA test.{" "}
          {targetEce != null && (
            <>
              Fitting on target data restores calibration: ECE ≈ <strong>{fmt(targetEce)}</strong> (Platt, disjoint source groups).
            </>
          )}
        </p>
      </Panel>

      <Panel title="Transfer robustness (B3)" subtitle="Zero-shot on external data — figures from artifacts/figures/b3.">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Figure src="/api/figures/b3/transfer_score_distributions.png" alt="Score distributions in-domain versus out-of-domain" className="w-full" />
          <Figure src="/api/figures/b3/context_length_robustness.png" alt="F1 by context length" className="w-full" />
        </div>
        {d.b3.transfer && (
          <div className="pt-2">
            <DataTable
              rowKey={(r) => String(r.subset)}
              columns={[
                { key: "subset", label: "Subset" },
                { key: "n_rows", label: "Rows", align: "right" },
                { key: "f1", label: "F1", align: "right" },
                { key: "auroc", label: "AUROC", align: "right" },
                { key: "delta", label: "ΔF1 vs in-domain", align: "right" },
              ]}
              rows={d.b3.transfer.map((t) => ({
                subset: t.subset,
                n_rows: t.n_rows,
                f1: fmt(t.f1),
                auroc: fmt(t.auroc),
                delta: `${t.delta_f1_vs_in_domain >= 0 ? "+" : ""}${fmt(t.delta_f1_vs_in_domain)}`,
              }))}
            />
          </div>
        )}
      </Panel>

      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Gauge className="w-6 h-6 text-violet-600 dark:text-purple-400" aria-hidden />
          <p className="text-xs text-muted-foreground">
            Live mode needs the local API (<code className="font-mono">uvicorn src.api.main:app</code>) — or the chat
            needs <code className="font-mono">OPENAI_API_KEY</code>. This page needs neither.
          </p>
        </div>
        <div className="flex gap-3">
          <Link href="/analyze" className="text-xs font-semibold bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white px-4 py-2 rounded-xl inline-flex items-center gap-2 transition-all">
            Try it live <ArrowRight className="w-3.5 h-3.5" aria-hidden />
          </Link>
          <Link href="/dashboard/overview" className="text-xs font-semibold bg-secondary/80 hover:bg-secondary border border-border px-4 py-2 rounded-xl transition-all">
            Full dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
