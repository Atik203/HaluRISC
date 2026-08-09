import { loadDashboardData, fmt } from "@/lib/results";
import { Panel, DataTable, EmptyState, Figure } from "@/components/dashboard/panel";

export default function FailuresTab() {
  const d = loadDashboardData();
  const b3Errors = d.b3.errorCases ?? [];
  const b5Failures = d.b5.failureCases ?? [];
  const b5Review = d.b5.reviewCases ?? [];
  const vaErrors = d.legacy.errorAnalysis;

  const failureByType = b5Failures.reduce<Record<string, number>>((acc, f) => {
    acc[f.perturbation] = (acc[f.perturbation] ?? 0) + 1;
    return acc;
  }, {});

  const redactedFaithbench = b3Errors.some((c) => c.source_dataset === "faithbench" && c.text_redacted);

  return (
    <div className="space-y-6">
      {!d.b3.errorCases && !d.b5.failureCases && !vaErrors && (
        <EmptyState message="No failure-case artifacts found (b3_error_cases.json / b5_failure_cases.json / error_analysis.json)." />
      )}

      {b5Failures.length > 0 && (
        <Panel
          title="Perturbation-induced failures (B5.6)"
          subtitle="Score moved by more than 0.3 or SHAP top-1 flipped under a controlled text edit."
        >
          <ul className="flex flex-wrap gap-3 text-xs">
            {Object.entries(failureByType).map(([type, n]) => (
              <li key={type} className="glass-panel px-3 py-2 rounded-xl flex items-center gap-2">
                <span className="text-muted-foreground">{type}</span>
                <span className="font-mono font-bold">{n}</span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-muted-foreground pt-1">
            {b5Failures.length} flagged evaluations from the B5 perturbation protocol — candidates for the manual review
            sheet, not automatic evidence of model failure.
          </p>
        </Panel>
      )}

      {b5Review.length > 0 && (
        <Panel
          title="Manual review sheet (B5.5)"
          subtitle="Sampled cases for two reviewers — fill reviewer_1 / reviewer_2 / agreement in b5_review_cases.csv."
        >
          <DataTable
            rowKey={(r) => String(r.sample_id)}
            columns={[
              { key: "sample_id", label: "ID" },
              { key: "label", label: "Label", align: "right" },
              { key: "raw", label: "Raw", align: "right" },
              { key: "cal", label: "Calibrated", align: "right" },
              { key: "top5", label: "Top-5 SHAP" },
              { key: "q", label: "Question (truncated)" },
            ]}
            rows={b5Review.slice(0, 20).map((r) => ({
              sample_id: r.sample_id,
              label: r.label,
              raw: fmt(r.raw_score),
              cal: fmt(r.calibrated_score),
              top5: r.top5_shap_features,
              q: r.question.length > 90 ? `${r.question.slice(0, 90)}…` : r.question,
            }))}
          />
        </Panel>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel
          title="B3 zero-shot error cases (b3_error_cases.json)"
          subtitle={redactedFaithbench ? "FaithBench text is redacted (CC BY-NC-SA) — join locally via sample_id." : "Sampled per subset."}
        >
          {b3Errors.length > 0 ? (
            <DataTable
              rowKey={(r) => `${r.source_dataset}-${r.sample_id}`}
              columns={[
                { key: "source_dataset", label: "Dataset" },
                { key: "task", label: "Task" },
                { key: "generator_model", label: "Generator" },
                { key: "label", label: "Label", align: "right" },
                { key: "raw_score", label: "Score", align: "right" },
                { key: "excerpt", label: "Answer (truncated)" },
              ]}
              rows={b3Errors.slice(0, 15).map((c) => ({
                sample_id: c.sample_id,
                source_dataset: c.source_dataset,
                task: c.task ?? "—",
                generator_model: c.generator_model ?? "—",
                label: c.label,
                raw_score: fmt(c.raw_score),
                excerpt:
                  c.text_redacted ?? (c.answer && c.answer.length > 80 ? `${c.answer.slice(0, 80)}…` : (c.answer ?? "—")),
              }))}
            />
          ) : (
            <p className="text-xs text-muted-foreground">b3_error_cases.json not present.</p>
          )}
        </Panel>

        <div className="space-y-6">
          <Panel title="Version A error analysis (legacy)" subtitle="Auto-tagged categories — review cases before paper use.">
            {vaErrors ? (
              <>
                <p className="text-xs text-muted-foreground mb-3">
                  {vaErrors.n_false_positives} FP / {vaErrors.n_false_negatives} FN of {vaErrors.n_test} test samples
                  (sampled {vaErrors.sampled?.fp ?? 0} FP / {vaErrors.sampled?.fn ?? 0} FN).
                </p>
                <DataTable
                  rowKey={(r) => String(r.category)}
                  columns={[
                    { key: "category", label: "Category" },
                    { key: "fp", label: "FP", align: "right" },
                    { key: "fn", label: "FN", align: "right" },
                  ]}
                  rows={Object.keys({
                    ...vaErrors.category_counts.false_positive,
                    ...vaErrors.category_counts.false_negative,
                  }).map((cat) => ({
                    category: cat,
                    fp: vaErrors.category_counts.false_positive[cat] ?? 0,
                    fn: vaErrors.category_counts.false_negative[cat] ?? 0,
                  }))}
                />
              </>
            ) : (
              <p className="text-xs text-muted-foreground">error_analysis.json not present.</p>
            )}
          </Panel>
          <Panel title="Borderline / SHAP waterfall figures (legacy)">
            <div className="grid grid-cols-1 gap-3">
              <Figure src="/api/figures/fig_shap_waterfall_borderline.png" alt="SHAP waterfall for a borderline case" className="w-full" />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
