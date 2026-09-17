import { loadDashboardData, fmt, fmtPct } from "@/lib/results";
import { Panel, DataTable, Figure, EmptyState, Mono } from "@/components/dashboard/panel";
import { TransferBars } from "@/components/dashboard/charts";

const DATASET_LABELS: Record<string, string> = {
  ragtruth_qa_test: "RAGTruth QA test",
  ragtruth_summarization: "RAGTruth summarization",
  ragtruth_data_to_text: "RAGTruth data-to-text",
  ragtruth_all: "RAGTruth (all tasks)",
  faithbench: "FaithBench",
  ragtruth_train: "RAGTruth train (covariate)",
};

export default function RobustnessTab() {
  const d = loadDashboardData();
  const metrics = d.b3.datasetMetrics;
  const transfer = d.b3.transfer ?? [];
  const inDomain = d.b2.comparison?.find((r) => r.model === "xgboost");

  const metricRows = metrics
    ? Object.entries(metrics)
        .filter(([, v]) => v != null)
        .map(([name, v]) => ({
          subset: DATASET_LABELS[name] ?? name,
          n_rows: v!.n_rows,
          n_groups: v!.n_groups,
          f1: fmt(v!.f1_mean),
          auroc: fmt(v!.auroc_mean),
          ece: fmt(v!.ece_mean),
          pred_pos: fmtPct(v!.predicted_positive_rate),
          label_pos: fmtPct(v!.label_positive_rate),
        }))
    : [];

  return (
    <div className="space-y-6">
      <Panel
        title="Zero-shot transfer — B2 XGBoost on external RAGTruth + FaithBench (B3)"
        subtitle="No training on external data; threshold fixed at 0.5; source-group bootstrap CIs (1,000 resamples)."
      >
        <DataTable
          rowKey={(r) => String(r.subset)}
          columns={[
            { key: "subset", label: "Subset" },
            { key: "n_rows", label: "Rows", align: "right" },
            { key: "n_groups", label: "Groups", align: "right" },
            { key: "f1", label: "F1", align: "right" },
            { key: "auroc", label: "AUROC", align: "right" },
            { key: "ece", label: "ECE", align: "right" },
            { key: "pred_pos", label: "Pred. pos", align: "right" },
            { key: "label_pos", label: "Label pos", align: "right" },
          ]}
          rows={metricRows}
        />
        <p className="text-xs text-muted-foreground">
          In-domain reference (B2 test): F1 {inDomain ? fmt(inDomain.f1_mean) : "—"} · AUROC{" "}
          {inDomain ? fmt(inDomain.auroc_mean) : "—"} — the transfer gap below is the core robustness finding.
        </p>
      </Panel>

      <Panel title="Transfer comparison (Δ vs in-domain)" subtitle="b3_transfer_comparison.csv">
        {transfer.length > 0 ? (
          <>
            <TransferBars
              data={transfer.map((t) => ({
                subset: t.subset,
                f1: t.f1,
                delta: t.delta_f1_vs_in_domain,
              }))}
            />
            <DataTable
              rowKey={(r) => String(r.subset)}
              columns={[
                { key: "subset", label: "Subset" },
                { key: "n_rows", label: "Rows", align: "right" },
                { key: "f1", label: "F1", align: "right" },
                { key: "auroc", label: "AUROC", align: "right" },
                { key: "delta_f1", label: "ΔF1", align: "right" },
                { key: "delta_auroc", label: "ΔAUROC", align: "right" },
              ]}
              rows={transfer.map((t) => ({
                subset: DATASET_LABELS[t.subset] ?? t.subset,
                n_rows: t.n_rows,
                f1: fmt(t.f1),
                auroc: fmt(t.auroc),
                delta_f1: `${t.delta_f1_vs_in_domain >= 0 ? "+" : ""}${fmt(t.delta_f1_vs_in_domain)}`,
                delta_auroc: `${t.delta_auroc_vs_in_domain >= 0 ? "+" : ""}${fmt(t.delta_auroc_vs_in_domain)}`,
              }))}
            />
          </>
        ) : (
          <p className="text-xs text-muted-foreground">b3_transfer_comparison.csv not present.</p>
        )}
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Subgroup metrics (rows ≥ 100, groups ≥ 20)" subtitle="b3_subgroup_metrics.csv — reported only above minimums.">
          {d.b3.subgroups ? (
            <DataTable
              rowKey={(r) => `${r.dimension}-${r.subgroup}`}
              columns={[
                { key: "dimension", label: "Dimension" },
                { key: "subgroup", label: "Subgroup" },
                { key: "n_rows", label: "Rows", align: "right" },
                { key: "f1", label: "F1", align: "right" },
                { key: "ece", label: "ECE", align: "right" },
                { key: "note", label: "Note" },
              ]}
              rows={d.b3.subgroups.map((r) => ({
                dimension: r.dimension,
                subgroup: String(r.subgroup),
                n_rows: r.n_rows,
                f1: r.reported ? fmt(r.f1) : "—",
                ece: r.reported ? fmt(r.ece) : "—",
                note: r.reported ? "" : String(r.reason ?? "below minimum"),
              }))}
            />
          ) : (
            <p className="text-xs text-muted-foreground">b3_subgroup_metrics.csv not present.</p>
          )}
        </Panel>

        <div className="space-y-6">
          <Panel title="Context-length robustness (figure)" subtitle="F1 by context length (words), RAGTruth.">
            <Figure src="/api/figures/b3/context_length_robustness.png" alt="F1 by context length" />
          </Panel>
          <Panel title="Bootstrap CIs + label sensitivity" subtitle="Source-group resampling, seed 777 (B3).">
            {d.b3.bootstrapCis ? (
              <ul className="space-y-1.5 text-xs">
                {Object.entries(d.b3.bootstrapCis).map(([name, v]) => (
                  <li key={name} className="flex justify-between gap-4">
                    <span className="text-muted-foreground">{DATASET_LABELS[name] ?? name}</span>
                    <span className="font-mono">
                      F1 [{v.f1_ci?.[0] != null ? fmt(v.f1_ci[0], 3) : "—"}, {v.f1_ci?.[1] != null ? fmt(v.f1_ci[1], 3) : "—"}] · AUROC [{v.auroc_ci?.[0] != null ? fmt(v.auroc_ci[0], 3) : "—"}, {v.auroc_ci?.[1] != null ? fmt(v.auroc_ci[1], 3) : "—"}]
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted-foreground">b3_bootstrap_cis.json not present.</p>
            )}
            {d.b3.labelSensitivity && (
              <div className="pt-2 border-t border-border/40">
                <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">FaithBench label-mapping sensitivity</h3>
                <pre className="text-[11px] font-mono overflow-auto max-h-48 rounded-xl bg-secondary/40 p-3 border border-border/50">
                  {JSON.stringify(d.b3.labelSensitivity, null, 2)}
                </pre>
              </div>
            )}
          </Panel>
        </div>
      </div>

      {!d.b3.datasetMetrics && (
        <EmptyState message="No B3 results found. Run the Version B pipeline to populate artifacts/results/b3/." />
      )}
    </div>
  );
}
