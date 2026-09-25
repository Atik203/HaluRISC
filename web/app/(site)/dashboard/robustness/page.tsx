import { loadDashboardData, fmt, fmtPct } from "@/lib/results";
import { Panel, DataTable, Figure, EmptyState } from "@/components/dashboard/panel";
import { TransferBars } from "@/components/dashboard/charts";

const LABEL_MAPPINGS: Record<string, string> = {
  primary_worst_q_plus_unwanted: "Primary (worst + question + unwanted)",
  majority_q_plus_unwanted: "Majority (question + unwanted)",
  strict_worst_unwanted_only: "Strict (unwanted only)",
};

const DATASET_LABELS: Record<string, string> = {
  ragtruth_qa_test: "RAGTruth QA test",
  ragtruth_summarization: "RAGTruth summarization",
  ragtruth_data_to_text: "RAGTruth data-to-text",
  ragtruth_all: "RAGTruth (all tasks)",
  faithbench: "FaithBench",
  ragtruth_train: "RAGTruth train (covariate)",
};

const EC_DATASETS: Record<string, string> = {
  ragtruth_all_test: "RAGTruth (all test tasks)",
  ragtruth_qa_test: "RAGTruth QA test (held out)",
  faithbench: "FaithBench",
};

const EC_VARIANTS: Record<string, string> = {
  m0: "Standard XGBoost",
  m1: "m1 debias + constraints",
  m2: "m2 + claim features",
  m3: "m3 EC-XGB (deployed)",
};

export default function RobustnessTab() {
  const d = loadDashboardData();
  const metrics = d.b3.datasetMetrics;
  const transfer = d.b3.transfer ?? [];
  const inDomain = d.b2.comparison?.find((r) => r.model === "xgboost");
  const ecExternal = d.b6.external ?? [];
  const ecComparison = d.b6.comparison ?? [];
  const ecStrictRuns = d.b6.strict ? Object.values(d.b6.strict.per_run) : [];
  const ecCalibration = d.b6.calibration;
  const ecComponents = d.b6.runConfig?.component_map ?? {};
  const strictMin = (pick: (r: (typeof ecStrictRuns)[number]) => number) =>
    ecStrictRuns.length > 0 ? Math.min(...ecStrictRuns.map(pick)) : null;
  const strictMax = (pick: (r: (typeof ecStrictRuns)[number]) => number) =>
    ecStrictRuns.length > 0 ? Math.max(...ecStrictRuns.map(pick)) : null;

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
        title="EC-XGB under domain shift (B6) — deployed model"
        subtitle="m3 variant (Evidence-Consistent XGBoost), seeds 42/123/456, threshold 0.5. Standard = B2 XGBoost trained on HaluEval only; EC-XGB adds the RAGTruth non-QA training rows. Flagged = share above 0.5."
      >
        {ecExternal.length > 0 ? (
          <DataTable
            rowKey={(r) => `${r.dataset}-${r.model}`}
            highlight={(r) => r.model.startsWith("m3")}
            columns={[
              { key: "dataset", label: "Corpus" },
              { key: "model", label: "Model" },
              { key: "n", label: "Rows", align: "right" },
              { key: "f1", label: "F1", align: "right" },
              { key: "auroc", label: "AUROC", align: "right" },
              { key: "flagged", label: "Flagged", align: "right" },
              { key: "ece", label: "ECE", align: "right" },
            ]}
            rows={["ragtruth_all_test", "ragtruth_qa_test", "faithbench"].flatMap((dataset) =>
              ["m0", "m3"].map((variant) => {
                const row = ecExternal.find((r) => r.dataset === dataset && r.variant === variant);
                return {
                  dataset: EC_DATASETS[dataset] ?? dataset,
                  model: EC_VARIANTS[variant] ?? variant,
                  n: row?.n ?? "—",
                  f1: fmt(row?.f1),
                  auroc: fmt(row?.auroc),
                  flagged: fmtPct(row?.predicted_positive_rate),
                  ece: fmt(row?.ece),
                };
              }),
            )}
          />
        ) : (
          <p className="text-xs text-muted-foreground">b6_external_metrics.csv not present.</p>
        )}
      </Panel>

      <Panel
        title="EC-XGB ablation, operating point, and display calibration (B6)"
        subtitle="b6_model_comparison.csv · b6_strict_mode.json · b6_calibration.json"
      >
        {ecComparison.length > 0 ? (
          <DataTable
            rowKey={(r) => r.variant}
            highlight={(r) => r.variant.startsWith("m3")}
            columns={[
              { key: "variant", label: "Variant" },
              { key: "n_features", label: "Features", align: "right" },
              { key: "f1", label: "F1", align: "right" },
              { key: "auroc", label: "AUROC", align: "right" },
              { key: "mcc", label: "MCC", align: "right" },
              { key: "ece", label: "ECE", align: "right" },
            ]}
            rows={ecComparison.map((r) => ({
              variant: EC_VARIANTS[r.variant] ?? ecComponents[r.variant] ?? r.variant,
              n_features: r.n_features ?? "—",
              f1: fmt(r.f1_mean),
              auroc: fmt(r.auroc_mean),
              mcc: fmt(r.mcc_mean),
              ece: fmt(r.ece_mean),
            }))}
          />
        ) : (
          <p className="text-xs text-muted-foreground">b6_model_comparison.csv not present.</p>
        )}
        <div className="mt-3 space-y-1.5 border-t border-border/40 pt-3 text-xs text-muted-foreground">
          {ecStrictRuns.length > 0 && (
            <p>
              Operating point at a {fmtPct(d.b6.strict?.alpha)} false-positive budget: test FPR{" "}
              {fmtPct(strictMin((r) => r.test_fpr))}–{fmtPct(strictMax((r) => r.test_fpr))} with recall{" "}
              {fmtPct(strictMin((r) => r.test_recall))}–{fmtPct(strictMax((r) => r.test_recall))}.
            </p>
          )}
          {ecCalibration && (
            <p>
              Deployed display score ({ecCalibration.chosen} on {ecCalibration.calibration_rows.toLocaleString()} RAGTruth
              QA calibration rows): held-out ECE {fmt(ecCalibration.test_raw.ece, 3)} → {fmt(ecCalibration.test_platt.ece, 3)}.
            </p>
          )}
        </div>
      </Panel>

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
              <div className="space-y-2 border-t border-border/40 pt-3">
                <h3 className="eyebrow">FaithBench label-mapping sensitivity</h3>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  The measured F1 depends on which FaithBench labels count as positive.
                </p>
                <DataTable
                  rowKey={(r) => String(r.mapping)}
                  columns={[
                    { key: "mapping", label: "Mapping" },
                    { key: "pos", label: "Pos.", align: "right" },
                    { key: "neg", label: "Neg.", align: "right" },
                    { key: "f1", label: "F1", align: "right" },
                    { key: "auroc", label: "AUROC", align: "right" },
                    { key: "mcc", label: "MCC", align: "right" },
                  ]}
                  rows={Object.entries(d.b3.labelSensitivity).map(([mapping, v]) => ({
                    mapping: LABEL_MAPPINGS[mapping] ?? mapping,
                    pos: v.n_positive,
                    neg: v.n_negative,
                    f1: fmt(v.f1),
                    auroc: fmt(v.auroc),
                    mcc: fmt(v.mcc),
                  }))}
                />
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
