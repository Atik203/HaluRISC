import { Award, DollarSign, GitCommit, ShieldCheck, TrendingUp, Zap } from "lucide-react";
import { loadDashboardData, fmt, fmtPct, modelVersionFromManifest } from "@/lib/results";
import { Panel, KpiCard, DataTable, Mono, EmptyState } from "@/components/dashboard/panel";

const MODEL_LABELS: Record<string, string> = {
  xgboost: "XGBoost (ours)",
  rf_full: "Random Forest",
  lr_full: "Logistic Regression",
  heuristic_overlap: "Heuristic (1 − overlap)",
  majority: "Majority",
  nli_only: "NLI-only",
  tfidf_context: "TF-IDF (context)",
  tfidf_answer: "TF-IDF (answer)",
  tfidf_all: "TF-IDF (all)",
};

export default function OverviewTab() {
  const d = loadDashboardData();
  const rows = d.b2.comparison ?? [];
  const stats = d.b2.statisticalTests;
  const m = d.manifest;
  const xgb = rows.find((r) => r.model === "xgboost");
  const bestBaseline = rows.find((r) => r.model === stats?.best_baseline);

  const b4Ece = d.b4.metrics?.["halueval_test"]?.platt?.ece_mean;
  const judgeCost = d.legacy.judge?.cost_per_1000_usd;
  const haluriscCost = d.legacy.latency?.cost_per_1000_predictions_usd?.halurisc_local;
  const costRatio = judgeCost && haluriscCost ? Math.round(judgeCost / haluriscCost) : null;

  return (
    <div className="space-y-6">
      {!xgb ? (
        <EmptyState message="No B2 results found. Run the Version B pipeline (see README 'Reproducibility') to populate artifacts/results/b2/." />
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard icon={<Award className="w-5 h-5" />} label="Test F1 (B2 XGBoost, seeds 42/123/456)" value={fmt(xgb.f1_mean)} accent="purple" sub={`± ${fmt(xgb.f1_std)} std`} />
            <KpiCard icon={<TrendingUp className="w-5 h-5" />} label="AUROC" value={fmt(xgb.auroc_mean)} accent="emerald" sub={`± ${fmt(xgb.auroc_std)} std`} />
            <KpiCard icon={<Zap className="w-5 h-5" />} label="ECE after Platt (HaluEval test, B4)" value={b4Ece != null ? b4Ece.toFixed(4) : "—"} accent="blue" />
            <KpiCard icon={<DollarSign className="w-5 h-5" />} label="Cheaper than LLM Judge (measured)" value={costRatio ? `${costRatio}×` : "—"} accent="amber" />
          </div>

          <Panel
            title="Corrected Baselines — grouped 5-fold CV, seeds 42/123/456 (B2)"
            subtitle="Grouped by item_idx (no leakage); McNemar and bootstrap CIs from b2_statistical_tests.json."
          >
            <DataTable
              rowKey={(r) => String(r.model)}
              highlight={(r) => r.model === "xgboost"}
              columns={[
                { key: "model", label: "Model" },
                { key: "deterministic", label: "Det." },
                { key: "precision_mean", label: "Precision", align: "right" },
                { key: "recall_mean", label: "Recall", align: "right" },
                { key: "f1_mean", label: "F1", align: "right" },
                { key: "auroc_mean", label: "AUROC", align: "right" },
                { key: "pr_auc_mean", label: "PR-AUC", align: "right" },
                { key: "mcc_mean", label: "MCC", align: "right" },
                { key: "ece_mean", label: "ECE", align: "right" },
              ]}
              rows={rows.map((r) => ({
                model: MODEL_LABELS[r.model] ?? r.model,
                deterministic: r.deterministic ? "yes" : "no",
                precision_mean: fmt(r.precision_mean),
                recall_mean: fmt(r.recall_mean),
                f1_mean: fmt(r.f1_mean),
                auroc_mean: fmt(r.auroc_mean),
                pr_auc_mean: fmt(r.pr_auc_mean),
                mcc_mean: fmt(r.mcc_mean),
                ece_mean: fmt(r.ece_mean),
              }))}
            />
            {stats?.mcnemar_vs_best_baseline != null && (
              <p className="text-xs text-muted-foreground">
                McNemar (XGBoost vs best baseline {stats.best_baseline ?? "—"}): p ={" "}
                {fmt(stats.mcnemar_vs_best_baseline)}
                {bestBaseline ? ` · best baseline F1 ${fmt(bestBaseline.f1_mean)}` : ""}
                {Array.isArray(stats.bootstrap_xgb_f1_ci) && (
                  <>
                    {" "}· Bootstrap 95% CI F1 [{fmt(stats.bootstrap_xgb_f1_ci[0], 3)}, {fmt(stats.bootstrap_xgb_f1_ci[1], 3)}]
                  </>
                )}
              </p>
            )}
          </Panel>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Panel title="Pipeline provenance (manifest.json)" subtitle="Frozen run metadata — commit, fingerprint, hardware.">
              <dl className="space-y-2 text-xs">
                {[
                  ["Model version", modelVersionFromManifest(m)],
                  ["Git commit", m?.git_commit?.slice(0, 12) ?? "— (Colab run)"],
                  ["Source fingerprint", m?.source_fingerprint?.slice(0, 16) ?? "—"],
                  ["Seeds", m?.seeds?.join(", ") ?? "—"],
                  ["GPU", m?.hardware?.cuda ? `${m.hardware.gpu_name} (${m.hardware.gpu_total_memory_gb} GB)` : "none"],
                  ["RAM", m?.hardware?.ram_total_gb != null ? `${m.hardware.ram_total_gb} GB` : "—"],
                  ["Split", m?.split_report?.leakage_free ? `leakage-free (${m.split_report.n_groups_total} groups)` : "—"],
                ].map(([k, v]) => (
                  <div key={String(k)} className="flex justify-between gap-4 border-b border-border/40 pb-1.5">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd className="font-mono text-right">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </Panel>

            <Panel title="Leakage-removal impact (B2)" subtitle="Historical leaked vs corrected grouped-split comparison.">
              {d.b2.leakageComparison ? (
                <pre className="text-[11px] font-mono overflow-auto max-h-64 rounded-xl bg-secondary/40 p-3 border border-border/50">
                  {JSON.stringify(d.b2.leakageComparison, null, 2)}
                </pre>
              ) : (
                <p className="text-xs text-muted-foreground">b2_leakage_comparison.json not present.</p>
              )}
              {d.b2.tuning && (
                <div className="pt-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">Tuning (per seed, 5-fold group CV)</h3>
                  <ul className="space-y-1 text-xs">
                    {Object.entries(d.b2.tuning).map(([seed, t]) => (
                      <li key={seed} className="flex justify-between gap-4">
                        <span className="font-mono text-muted-foreground">seed {seed}</span>
                        <span className="font-mono">cv_auc {fmt(t.best_cv_auc, 4)} · lr {t.best_params.learning_rate} · depth {t.best_params.max_depth} · n {t.best_params.n_estimators}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Panel>
          </div>

          <p className="text-[11px] text-muted-foreground flex items-center gap-2">
            <ShieldCheck className="w-4 h-4" aria-hidden />
            Every value above is read from artifacts/results/b2/* at request time. ECE/calibration details live in the
            Calibration tab; transfer evidence in the Robustness tab.
          </p>
        </>
      )}
    </div>
  );
}
