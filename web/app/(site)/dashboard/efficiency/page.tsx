import { loadDashboardData, fmt } from "@/lib/results";
import { Panel, DataTable, EmptyState } from "@/components/dashboard/panel";

const MODULE_LABELS: Record<string, string> = {
  core_lexical: "Lexical",
  entity: "Entity",
  nli: "NLI",
  semantic: "Semantic",
  model: "XGBoost",
  shap: "SHAP",
};

export default function EfficiencyTab() {
  const d = loadDashboardData();
  const latency = d.legacy.latency;
  const judge = d.legacy.judge;
  const latencyMs = latency?.latency_ms ?? {};
  const modelArtifactMb = latency?.model_artifact_mb;
  const modelMb =
    typeof modelArtifactMb === "number"
      ? modelArtifactMb
      : modelArtifactMb
        ? Object.values(modelArtifactMb).reduce((a, b) => a + (b ?? 0), 0)
        : null;

  const moduleRows = Object.entries(latencyMs).map(([name, v]) => ({
    module: MODULE_LABELS[name] ?? name,
    p50: v.p50,
    p95: v.p95,
    mean: v.mean,
  }));

  const haluriscCost = latency?.cost_per_1000_predictions_usd?.halurisc_local;
  const judgeCost =
    judge?.cost_per_1000_usd ?? latency?.cost_per_1000_predictions_usd?.llm_judge_estimate;

  return (
    <div className="space-y-6">
      {!latency && !judge && (
        <EmptyState message="No efficiency artifacts found (latency_analysis.json / llm_judge_results.json)." />
      )}

      {latency && (
        <Panel title="Per-module latency (measured)" subtitle="latency_analysis.json — p50 / p95 / mean milliseconds per sample.">
          <DataTable
            rowKey={(r) => String(r.module)}
            columns={[
              { key: "module", label: "Module" },
              { key: "p50", label: "p50 (ms)", align: "right" },
              { key: "p95", label: "p95 (ms)", align: "right" },
              { key: "mean", label: "Mean (ms)", align: "right" },
            ]}
            rows={moduleRows.map((r) => ({
              module: r.module,
              p50: r.p50 != null ? r.p50.toFixed(1) : "—",
              p95: r.p95 != null ? r.p95.toFixed(1) : "—",
              mean: r.mean != null ? r.mean.toFixed(1) : "—",
            }))}
          />
          <p className="text-xs text-muted-foreground">
            Total per sample: p50 {latency.total_per_sample_ms?.p50 != null ? `${latency.total_per_sample_ms.p50.toFixed(0)} ms` : "—"}
            {modelMb != null && ` · model artifacts ≈ ${modelMb} MB`}
          </p>
        </Panel>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Cost per 1,000 predictions (measured)" subtitle="From latency/cost artifacts — never hardcoded.">
          <dl className="space-y-2 text-xs">
            <div className="flex justify-between gap-4 border-b border-border/40 pb-1.5">
              <dt className="text-muted-foreground">HaluRISC (local CPU/GPU)</dt>
              <dd className="font-mono">{haluriscCost != null ? `$${haluriscCost.toFixed(4)}` : "—"}</dd>
            </div>
            <div className="flex justify-between gap-4 border-b border-border/40 pb-1.5">
              <dt className="text-muted-foreground">LLM-as-Judge estimate</dt>
              <dd className="font-mono">{judgeCost != null ? `$${judgeCost.toFixed(4)}` : "—"}</dd>
            </div>
            {haluriscCost && judgeCost && (
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground font-semibold">Ratio</dt>
                <dd className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  {Math.round(judgeCost / haluriscCost)}× cheaper
                </dd>
              </div>
            )}
          </dl>
        </Panel>

        <Panel
          title={judge ? `LLM-as-Judge vs XGBoost (${judge.n_samples} samples, ${judge.model})` : "LLM-as-Judge comparison"}
          subtitle="llm_judge_results.json — agreement and cost measured on the same subset."
        >
          {judge ? (
            <DataTable
              rowKey={(r) => String(r.model)}
              columns={[
                { key: "model", label: "Model" },
                { key: "accuracy", label: "Accuracy", align: "right" },
                { key: "f1", label: "F1", align: "right" },
                { key: "p50", label: "Latency p50 (ms)", align: "right" },
              ]}
              rows={[
                {
                  model: `${judge.model} judge`,
                  accuracy: fmt(judge.judge.accuracy, 3),
                  f1: fmt(judge.judge.f1, 3),
                  p50: judge.judge.latency_ms_p50.toFixed(0),
                },
                {
                  model: "XGBoost (ours)",
                  accuracy: fmt(judge.xgboost_on_same_subset.accuracy, 3),
                  f1: fmt(judge.xgboost_on_same_subset.f1, 3),
                  p50: latency?.total_per_sample_ms?.p50 != null ? latency.total_per_sample_ms.p50.toFixed(0) : "—",
                },
              ]}
            />
          ) : (
            <p className="text-xs text-muted-foreground">
              Run src/models/eval_llm_judge.py to populate (measured cost lands in llm_judge_results.json).
            </p>
          )}
          {judge && (
            <p className="text-xs text-muted-foreground">
              Agreement with XGBoost: {fmt(judge.agreement_with_xgboost, 3)}
              {(judge.mcnemar_judge_vs_xgboost_p ?? judge.mcnemar_p) != null && (
                <> · McNemar p = {fmt(judge.mcnemar_judge_vs_xgboost_p ?? judge.mcnemar_p)}</>
              )}
              {judge.cost_usd != null && ` · judge run cost $${judge.cost_usd.toFixed(4)}`}
            </p>
          )}
        </Panel>
      </div>
    </div>
  );
}
