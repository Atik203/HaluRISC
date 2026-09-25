import { loadDashboardData, fmt, fmtPct } from "@/lib/results";
import { Panel, DataTable, EmptyState } from "@/components/dashboard/panel";
import {
  GroupImportanceBars,
  NeutralizationChart,
  PerturbationChart,
  StabilityBars,
} from "@/components/dashboard/charts";

export default function ExplainabilityTab() {
  const d = loadDashboardData();
  const imp = d.b5.importance;
  const neu = d.b5.neutralization ?? [];
  const stab = d.b5.stability;
  const pert = d.b5.perturbations ?? [];

  const groupRows = imp
    ? Object.keys({ ...imp.group_mean_abs_shap, ...imp.group_ablation_f1_delta }).map((g) => ({
        group: g,
        shap: imp.group_mean_abs_shap[g] ?? 0,
        ablation: imp.group_ablation_f1_delta[g] ?? 0,
      }))
    : [];

  const topFeatures = imp ? Object.entries(imp.mean_abs_shap).slice(0, 12) : [];
  const stabilityRows = stab
    ? Object.entries(stab.feature_mean_abs_shap_ci)
        .sort((a, b) => b[1].mean - a[1].mean)
        .slice(0, 10)
        .map(([f, v]) => ({ feature: f, mean: v.mean, lo: v.lo, hi: v.hi }))
    : [];

  return (
    <div className="space-y-6">
      {!imp && <EmptyState message="No B5 results found. Run the Version B pipeline to populate artifacts/results/b5/." />}

      {imp && (
        <>
          <Panel
            title="Importance triangulation (B5)"
            subtitle="Mean |SHAP| vs permutation importance (test split, B2 seed-42 model); group ablation via neutralization proxy."
          >
            <p className="text-xs text-muted-foreground">
              Kendall τ (SHAP ranking vs permutation importance):{" "}
              <strong className="font-mono">{fmt(imp.kendall_tau_shap_vs_permutation)}</strong>
            </p>
            <GroupImportanceBars data={groupRows} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">Top features (mean |SHAP|)</h3>
                <ul className="space-y-1.5 text-xs">
                  {topFeatures.map(([f, v], i) => (
                    <li key={f} className="flex items-center gap-3">
                      <span className="w-5 text-muted-foreground font-mono">{i + 1}</span>
                      <span className="flex-1 truncate">{f}</span>
                      <span className="font-mono text-purple-700 dark:text-purple-300">{fmt(v)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">Group ablation ΔF1 (neutralization proxy)</h3>
                <ul className="space-y-1.5 text-xs">
                  {groupRows
                    .slice()
                    .sort((a, b) => b.ablation - a.ablation)
                    .map((g) => (
                      <li key={g.group} className="flex items-center gap-3">
                        <span className="flex-1 truncate">{g.group}</span>
                        <span className={`font-mono ${g.ablation > 0.02 ? "text-rose-600 dark:text-rose-400" : "text-muted-foreground"}`}>
                          {g.ablation >= 0 ? "+" : ""}
                          {fmt(g.ablation, 5)}
                        </span>
                      </li>
                    ))}
                </ul>
                <p className="text-[11px] text-muted-foreground mt-2">
                  Retrain-based ablation lives in ablation_results.csv (Version A); B5 reports the neutralization proxy
                  (roadmap B5.1).
                </p>
              </div>
            </div>
          </Panel>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Panel title="Neutralization curve (B5.2)" subtitle="Top-k features set to their test median — prediction change.">
              <NeutralizationChart data={neu.map((r) => ({ k: r.k, mean_score_delta: r.mean_score_delta }))} />
              <DataTable
                rowKey={(r) => String(r.k)}
                columns={[
                  { key: "k", label: "k", align: "right" },
                  { key: "top", label: "Top features" },
                  { key: "delta", label: "Mean Δscore", align: "right" },
                  { key: "f1", label: "ΔF1", align: "right" },
                ]}
                rows={neu.map((r) => ({
                  k: r.k,
                  top: r.top_features.join(", "),
                  delta: `${r.mean_score_delta >= 0 ? "+" : ""}${fmt(r.mean_score_delta)}`,
                  f1: `${r.f1_delta >= 0 ? "+" : ""}${fmt(r.f1_delta)}`,
                }))}
              />
            </Panel>

            <Panel title="Perturbation stability (B5.3)" subtitle="Full feature re-extraction after text edits; no fixed thresholds.">
              <PerturbationChart
                data={pert.map((p) => ({
                  perturbation: p.perturbation,
                  mean_abs_score_delta: p.mean_abs_score_delta,
                  top1_flip_rate: p.top1_flip_rate,
                }))}
              />
              <DataTable
                rowKey={(r) => String(r.perturbation)}
                columns={[
                  { key: "perturbation", label: "Perturbation" },
                  { key: "n", label: "n", align: "right" },
                  { key: "mean", label: "Mean |Δscore|", align: "right" },
                  { key: "large", label: "|Δ| > 0.3 rate", align: "right" },
                  { key: "flip", label: "Top-1 flip", align: "right" },
                  { key: "spearman", label: "SHAP ρ", align: "right" },
                ]}
                rows={pert.map((p) => ({
                  perturbation: p.perturbation,
                  n: p.n,
                  mean: fmt(p.mean_abs_score_delta),
                  large: fmtPct(p.large_delta_rate),
                  flip: fmtPct(p.top1_flip_rate),
                  spearman: fmt(p.mean_spearman),
                }))}
              />
            </Panel>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Panel title="Bootstrap stability (B5.4)" subtitle="1,000 resamples; top-k set Jaccard + mean-|SHAP| envelope.">
              {stab && (
                <>
                  <p className="text-xs text-muted-foreground mb-3">
                    Top-{stab.top_k} feature set Jaccard: <strong className="font-mono">{fmt(stab.topk_set_jaccard.mean, 3)} ± {fmt(stab.topk_set_jaccard.std, 3)}</strong>
                  </p>
                  <StabilityBars data={stabilityRows} />
                </>
              )}
            </Panel>

            <Panel title="Reviewer export (B5.5)" subtitle="40-case manual plausibility sheet — two reviewers, disagreements recorded.">
              {d.b5.reviewCases ? (
                <>
                  <p className="text-xs text-muted-foreground">
                    {d.b5.reviewCases.length} cases exported (10 FP / 10 FN / 20 borderline) to b5_review_cases.csv with
                    reviewer_1 / reviewer_2 / agreement columns to fill.
                  </p>
                  <DataTable
                    rowKey={(r) => String(r.sample_id)}
                    columns={[
                      { key: "sample_id", label: "ID" },
                      { key: "label", label: "Label", align: "right" },
                      { key: "raw", label: "Raw", align: "right" },
                      { key: "cal", label: "Calibrated", align: "right" },
                      { key: "top5", label: "Top-5 SHAP features" },
                    ]}
                    rows={d.b5.reviewCases.slice(0, 12).map((r) => ({
                      sample_id: r.sample_id,
                      label: r.label,
                      raw: fmt(r.raw_score),
                      cal: fmt(r.calibrated_score),
                      top5: r.top5_shap_features,
                    }))}
                  />
                </>
              ) : (
                <p className="text-xs text-muted-foreground">b5_review_cases.json not present.</p>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
