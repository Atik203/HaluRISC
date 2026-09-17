import { loadDashboardData, fmt } from "@/lib/results";
import { Panel, DataTable, Figure, EmptyState } from "@/components/dashboard/panel";
import { CalibrationBars } from "@/components/dashboard/charts";

const SUBSET_LABELS: Record<string, string> = {
  halueval_test: "HaluEval test (in-domain)",
  ragtruth_qa_test: "RAGTruth QA test",
  ragtruth_summarization: "RAGTruth summarization",
  ragtruth_data_to_text: "RAGTruth data-to-text",
  ragtruth_all: "RAGTruth (all)",
  faithbench: "FaithBench",
};

export default function CalibrationTab() {
  const d = loadDashboardData();
  const metrics = d.b4.metrics;
  const target = d.b4.target;

  const barData = metrics
    ? Object.entries(metrics)
        .filter(([, methods]) => methods?.raw && methods.platt)
        .map(([name, methods]) => ({
          subset: SUBSET_LABELS[name] ?? name,
          raw: methods.raw?.ece_mean ?? null,
          platt: methods.platt?.ece_mean ?? null,
          isotonic: methods.isotonic?.ece_mean ?? null,
        }))
    : [];

  const rows = metrics
    ? Object.entries(metrics)
        .filter(([, methods]) => methods != null)
        .flatMap(([name, methods]) =>
          Object.entries(methods!).map(([method, v]) => ({
            subset: SUBSET_LABELS[name] ?? name,
            method,
            ece: fmt(v.ece_mean),
            ace: fmt(v.ace_mean),
            brier: fmt(v.brier_mean),
            nll: fmt(v.nll_mean),
            slope: fmt(v.slope_mean),
            f1: fmt(v.f1_mean),
            auroc: fmt(v.auroc_mean),
          })),
        )
    : [];

  const targetRows = target
    ? [
        { label: "Raw source scores", ...(target.methods.raw_reference ?? {}) },
        { label: "Source-calibrated (Platt)", ...(target.methods.source_platt_reference ?? {}) },
        { label: "Source-calibrated (isotonic)", ...(target.methods.source_isotonic_reference ?? {}) },
        { label: "Target-calibrated (Platt)", ...(target.methods.platt ?? {}) },
        { label: "Target-calibrated (isotonic)", ...(target.methods.isotonic ?? {}) },
      ]
    : [];

  return (
    <div className="space-y-6">
      {!metrics && <EmptyState message="No B4 results found. Run the Version B pipeline to populate artifacts/results/b4/." />}

      {metrics && (
        <>
          <Panel
            title="Calibration under distribution shift (B4)"
            subtitle="Source calibrators fit on HaluEval validation only (seeds 42/123/456); applied unchanged to external sets."
          >
            <CalibrationBars data={barData} />
            <p className="text-xs text-muted-foreground">
              Predeclared deployable calibrator: <strong>Platt</strong> (isotonic reported for comparison, roadmap B4.2).
            </p>
          </Panel>

          <Panel title="Full metric table — ECE / ACE / Brier / NLL per subset × method">
            <DataTable
              rowKey={(r) => `${r.subset}-${r.method}`}
              highlight={(r) => r.method === "platt"}
              columns={[
                { key: "subset", label: "Subset" },
                { key: "method", label: "Method" },
                { key: "ece", label: "ECE", align: "right" },
                { key: "ace", label: "ACE", align: "right" },
                { key: "brier", label: "Brier", align: "right" },
                { key: "nll", label: "NLL", align: "right" },
                { key: "slope", label: "Slope", align: "right" },
                { key: "f1", label: "F1", align: "right" },
                { key: "auroc", label: "AUROC", align: "right" },
              ]}
              rows={rows}
            />
          </Panel>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Panel title="Target calibration — RAGTruth QA train → test" subtitle="Disjoint source groups; the headline shift fix.">
              {target && (
                <>
                  <p className="text-xs text-muted-foreground mb-3">
                    {target.n_calibration_rows.toLocaleString()} calibration rows · {target.n_test_rows.toLocaleString()} test
                    rows · {target.overlapping_groups_removed} overlapping groups removed.
                  </p>
                  <DataTable
                    rowKey={(r) => String(r.label)}
                    columns={[
                      { key: "label", label: "Configuration" },
                      { key: "ece", label: "ECE", align: "right" },
                      { key: "brier", label: "Brier", align: "right" },
                      { key: "nll", label: "NLL", align: "right" },
                    ]}
                    rows={targetRows.map((r) => ({
                      label: r.label,
                      ece: r.ece_mean != null ? fmt(r.ece_mean) : "—",
                      brier: r.brier_mean != null ? fmt(r.brier_mean) : "—",
                      nll: r.nll_mean != null ? fmt(r.nll_mean) : "—",
                    }))}
                  />
                  <p className="text-xs text-muted-foreground mt-3">
                    Source calibration does <strong>not</strong> transfer (ECE ≈ 0.81 on QA test); fitting on target data
                    restores calibration (ECE ≈ 0.13).
                  </p>
                </>
              )}
            </Panel>

            <div className="space-y-6">
              <Panel title="Reliability diagrams (figure)">
                <Figure src="/api/figures/b4/reliability_diagrams.png" alt="Reliability diagrams" />
              </Panel>
              <Panel title="Calibration shift (figure)">
                <Figure src="/api/figures/b4/calibration_shift.png" alt="Calibration shift bar chart" />
              </Panel>
            </div>
          </div>

          <Panel title="Subgroup calibration (rows ≥ 100, groups ≥ 20)" subtitle="b4_subgroup_calibration.csv — pooled fallback below minimums.">
            {d.b4.subgroups ? (
              <DataTable
                rowKey={(r) => `${r.dimension}-${r.subgroup}`}
                columns={[
                  { key: "dimension", label: "Dimension" },
                  { key: "subgroup", label: "Subgroup" },
                  { key: "n_rows", label: "Rows", align: "right" },
                  { key: "raw_ece", label: "Raw ECE", align: "right" },
                  { key: "platt_ece", label: "Platt ECE", align: "right" },
                  { key: "isotonic_ece", label: "Isotonic ECE", align: "right" },
                  { key: "note", label: "Note" },
                ]}
                rows={d.b4.subgroups.map((r) => ({
                  dimension: r.dimension,
                  subgroup: String(r.subgroup),
                  n_rows: r.n_rows,
                  raw_ece: r.reported ? fmt(r.raw_ece) : "—",
                  platt_ece: r.reported ? fmt(r.platt_ece) : "—",
                  isotonic_ece: r.reported ? fmt(r.isotonic_ece) : "—",
                  note: r.reported ? "" : String(r.reason ?? "below minimum"),
                }))}
              />
            ) : (
              <p className="text-xs text-muted-foreground">b4_subgroup_calibration.csv not present.</p>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
