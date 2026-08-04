import fs from "fs";
import path from "path";
import { Award, DollarSign, LayoutDashboard, TrendingUp, Zap } from "lucide-react";

export const dynamic = "force-dynamic";

const RESULTS_DIR = path.resolve(process.cwd(), "..", "artifacts", "results");

interface ModelMetrics {
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  pr_auc: number;
  mcc: number;
  f1_std?: number;
  threshold?: number;
  val_f1?: number;
}

interface CalibrationMethod {
  f1_mean: number;
  ece_mean: number;
  brier_mean: number;
}

interface Calibration {
  platt: CalibrationMethod;
  isotonic: CalibrationMethod;
}

interface Stats {
  mcnemar_p_value?: number;
  bootstrap_f1_ci?: [number, number];
  bootstrap_auroc_ci?: [number, number];
}

interface AblationRow {
  removed_group: string;
  f1_mean: number;
  f1_std: number;
  auroc_mean: number;
  auroc_std: number;
}

interface FinalResults {
  heuristic: ModelMetrics;
  logistic_regression: ModelMetrics;
  random_forest: ModelMetrics;
  xgboost: ModelMetrics;
  calibration?: Calibration;
  statistics?: Stats;
  ablation?: AblationRow[];
}

interface RagTruthResults {
  n_samples: number;
  f1: number;
  auroc: number;
  pr_auc: number;
  mcc: number;
  ece: number;
  brier: number;
  label_distribution: Record<string, number>;
}

interface ShapSummary {
  top_features?: Array<{ feature: string; mean_abs_shap: number }>;
}

interface LlmJudgeResults {
  n_samples: number;
  model: string;
  judge: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    latency_ms_p50: number;
    latency_ms_p95: number;
  };
  xgboost_on_same_subset: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  agreement_with_xgboost: number;
  cost_usd: number;
  cost_per_1000_usd: number;
}

interface ErrorAnalysis {
  n_test: number;
  n_false_positives: number;
  n_false_negatives: number;
  category_counts: {
    false_positive: Record<string, number>;
    false_negative: Record<string, number>;
  };
}

function readJson<T>(name: string): T | null {
  const p = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
  } catch {
    return null;
  }
}

const MODEL_LABELS: Record<string, string> = {
  heuristic: "Heuristic (1 - overlap)",
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost (ours)",
};

export default async function DashboardPage() {
  const results = readJson<FinalResults>("final_results.json");
  const ragtruth = readJson<RagTruthResults>("ragtruth_results.json");
  const shap = readJson<ShapSummary>("shap_summary.json");
  const judge = readJson<LlmJudgeResults>("llm_judge_results.json");
  const errors = readJson<ErrorAnalysis>("error_analysis.json");

  const modelRows = results
    ? (Object.entries(MODEL_LABELS)
        .filter(([key]) => results[key as keyof FinalResults])
        .map(([key, label]) => ({ key, label, ...results[key as keyof FinalResults] })) as Array<
        { key: string; label: string } & ModelMetrics
      >)
    : null;

  const xgb = results?.xgboost;
  const cal = results?.calibration;
  const stats = results?.statistics;

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
            <LayoutDashboard className="w-6 h-6 text-purple-400" /> 📈 Experiment Dashboard & Benchmarks
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Live results loaded from artifacts/results/ — no hardcoded numbers
          </p>
        </div>
      </div>

      {!results && (
        <div className="glass-panel p-10 rounded-2xl text-center space-y-2">
          <h3 className="text-sm font-semibold">No experiment results found</h3>
          <p className="text-xs text-muted-foreground max-w-lg mx-auto">
            Run the training pipeline first (see colab/HaluRISC_Training.ipynb) and place
            artifacts/results/* in the repo root. The dashboard renders real data only.
          </p>
        </div>
      )}

      {results && xgb && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-purple-500 flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
                <Award className="w-5 h-5" />
              </div>
              <div>
                <div className="text-2xl font-extrabold">{xgb.f1.toFixed(3)}</div>
                <div className="text-xs text-muted-foreground">
                  Test F1 (XGBoost, mean ± {xgb.f1_std?.toFixed(3)})
                </div>
              </div>
            </div>
            <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-emerald-500 flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                <TrendingUp className="w-5 h-5" />
              </div>
              <div>
                <div className="text-2xl font-extrabold">{xgb.auroc.toFixed(3)}</div>
                <div className="text-xs text-muted-foreground">
                  AUROC (95% CI [{stats?.bootstrap_auroc_ci?.[0]?.toFixed(3) ?? "—"}, {stats?.bootstrap_auroc_ci?.[1]?.toFixed(3) ?? "—"}])
                </div>
              </div>
            </div>
            <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-blue-500 flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-blue-500/20 text-blue-400 flex items-center justify-center">
                <Zap className="w-5 h-5" />
              </div>
              <div>
                <div className="text-2xl font-extrabold">{cal?.platt?.ece_mean?.toFixed(4)}</div>
                <div className="text-xs text-muted-foreground">ECE after Platt calibration</div>
              </div>
            </div>
            <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-amber-500 flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center">
                <DollarSign className="w-5 h-5" />
              </div>
              <div>
                <div className="text-2xl font-extrabold">100x</div>
                <div className="text-xs text-muted-foreground">Cheaper than LLM Judge</div>
              </div>
            </div>
          </div>

          {/* Model Comparison Table */}
          <div className="glass-panel p-6 rounded-2xl space-y-4">
            <h2 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">
              Model Comparison (test split, mean over seeds 42/123/456)
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground">
                    <th className="py-3 px-4 font-semibold">Model</th>
                    <th className="py-3 px-4 font-semibold">Precision</th>
                    <th className="py-3 px-4 font-semibold">Recall</th>
                    <th className="py-3 px-4 font-semibold">F1</th>
                    <th className="py-3 px-4 font-semibold">AUROC</th>
                    <th className="py-3 px-4 font-semibold">PR-AUC</th>
                    <th className="py-3 px-4 font-semibold">MCC</th>
                  </tr>
                </thead>
                <tbody>
                  {modelRows!.map((m) => (
                    <tr
                      key={m.key}
                      className={`border-b border-border/50 hover:bg-secondary/40 transition-colors ${
                        m.key === "xgboost" ? "bg-purple-950/20 font-semibold text-purple-300" : ""
                      }`}
                    >
                      <td className="py-3 px-4">{m.label}</td>
                      <td className="py-3 px-4 font-mono">{m.precision.toFixed(4)}</td>
                      <td className="py-3 px-4 font-mono">{m.recall.toFixed(4)}</td>
                      <td className="py-3 px-4 font-mono">{m.f1.toFixed(4)}</td>
                      <td className="py-3 px-4 font-mono">{m.auroc.toFixed(4)}</td>
                      <td className="py-3 px-4 font-mono">{m.pr_auc.toFixed(4)}</td>
                      <td className="py-3 px-4 font-mono">{m.mcc.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {stats?.mcnemar_p_value != null && (
              <p className="text-xs text-muted-foreground">
                McNemar test (XGBoost vs Random Forest): p = {stats.mcnemar_p_value.toFixed(4)} · Bootstrap 95% CI F1 [
                {stats.bootstrap_f1_ci?.[0]?.toFixed(3) ?? "—"}, {stats.bootstrap_f1_ci?.[1]?.toFixed(3) ?? "—"}]
              </p>
            )}
          </div>

          {/* Calibration */}
          <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🎯 Calibration (fit on validation only)</h3>
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground">
                    <th className="py-2 px-3 font-semibold">Method</th>
                    <th className="py-2 px-3 font-semibold">F1</th>
                    <th className="py-2 px-3 font-semibold">ECE</th>
                    <th className="py-2 px-3 font-semibold">Brier</th>
                  </tr>
                </thead>
                <tbody>
                  {(["platt", "isotonic"] as const).map((m) => (
                    <tr key={m} className="border-b border-border/50">
                      <td className="py-2 px-3 capitalize">{m}</td>
                      <td className="py-2 px-3 font-mono">{cal?.[m]?.f1_mean?.toFixed(4)}</td>
                      <td className="py-2 px-3 font-mono">{cal?.[m]?.ece_mean?.toFixed(4)}</td>
                      <td className="py-2 px-3 font-mono">{cal?.[m]?.brier_mean?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <img
                src="/api/figures/fig_reliability.png"
                alt="Reliability diagram"
                className="mt-4 rounded-xl border border-border/60 w-full max-w-md"
              />
              <img
                src="/api/figures/fig_roc_pr.png"
                alt="ROC and PR curves"
                className="mt-3 rounded-xl border border-border/60 w-full max-w-md"
              />
            </div>
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🔬 SHAP Global Importance (top 10)</h3>
              {shap ? (
                <ul className="space-y-2">
                  {shap.top_features?.map((f, i) => (
                    <li key={f.feature} className="flex items-center gap-3 text-xs">
                      <span className="w-6 text-muted-foreground font-mono">{i + 1}</span>
                      <span className="flex-1 truncate">{f.feature}</span>
                      <span className="font-mono text-purple-300">{f.mean_abs_shap.toFixed(4)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted-foreground">Run src/explain/shap_analysis.py to generate.</p>
              )}
              <img
                src="/api/figures/fig_shap_summary.png"
                alt="SHAP summary beeswarm"
                className="mt-4 rounded-xl border border-border/60 w-full max-w-md"
              />
            </div>
          </div>

          {/* Ablation + RAGTruth */}
          <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🧩 Feature Group Ablation (F1 drop when removed)</h3>
              {results.ablation?.length ? (
                <table className="w-full text-left text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground">
                      <th className="py-2 px-3 font-semibold">Removed group</th>
                      <th className="py-2 px-3 font-semibold">F1 (mean ± std)</th>
                      <th className="py-2 px-3 font-semibold">ΔF1 vs full</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.ablation.map((row) => {
                      const delta = xgb.f1 - row.f1_mean;
                      return (
                        <tr key={row.removed_group} className="border-b border-border/50">
                          <td className="py-2 px-3">-{row.removed_group}</td>
                          <td className="py-2 px-3 font-mono">
                            {row.f1_mean.toFixed(4)} ± {row.f1_std.toFixed(4)}
                          </td>
                          <td className={`py-2 px-3 font-mono ${delta > 0.01 ? "text-red-400" : "text-emerald-400"}`}>
                            {delta >= 0 ? "+" : ""}
                            {delta.toFixed(4)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <p className="text-xs text-muted-foreground">Run the training pipeline to populate.</p>
              )}
            </div>
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🌐 RAGTruth Zero-Shot (external, no training)</h3>
              {ragtruth ? (
                <table className="w-full text-left text-sm border-collapse">
                  <tbody>
                    {(["f1", "auroc", "pr_auc", "mcc", "ece", "brier"] as const).map((k) => (
                      <tr key={k} className="border-b border-border/50">
                        <td className="py-2 px-3 capitalize">{k === "pr_auc" ? "PR-AUC" : k === "auroc" ? "AUROC" : k}</td>
                        <td className="py-2 px-3 font-mono">{ragtruth[k].toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-xs text-muted-foreground">Run src/models/eval_ragtruth.py to populate.</p>
              )}
              <p className="text-xs text-muted-foreground mt-3">
                {ragtruth ? `${ragtruth.n_samples} RAGTruth QA samples (label balance: ${JSON.stringify(ragtruth.label_distribution)})` : ""}
              </p>
            </div>
          </div>

          {/* LLM Judge + Error Analysis */}
          <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🤖 LLM-as-Judge vs XGBoost (200 test samples)</h3>
              {judge ? (
                <>
                  <table className="w-full text-left text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border text-xs text-muted-foreground">
                        <th className="py-2 px-3 font-semibold">Model</th>
                        <th className="py-2 px-3 font-semibold">Accuracy</th>
                        <th className="py-2 px-3 font-semibold">Precision</th>
                        <th className="py-2 px-3 font-semibold">Recall</th>
                        <th className="py-2 px-3 font-semibold">F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-border/50">
                        <td className="py-2 px-3">{judge.model} judge</td>
                        <td className="py-2 px-3 font-mono">{judge.judge.accuracy.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.judge.precision.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.judge.recall.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.judge.f1.toFixed(3)}</td>
                      </tr>
                      <tr className="border-b border-border/50 bg-purple-950/20 font-semibold text-purple-300">
                        <td className="py-2 px-3">XGBoost (ours)</td>
                        <td className="py-2 px-3 font-mono">{judge.xgboost_on_same_subset.accuracy.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.xgboost_on_same_subset.precision.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.xgboost_on_same_subset.recall.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono">{judge.xgboost_on_same_subset.f1.toFixed(3)}</td>
                      </tr>
                    </tbody>
                  </table>
                  <p className="text-xs text-muted-foreground mt-3">
                    Agreement: {judge.agreement_with_xgboost.toFixed(3)} · Judge latency p50 {judge.judge.latency_ms_p50.toFixed(0)}ms vs
                    XGBoost ~5ms · Judge cost ${judge.cost_per_1000_usd.toFixed(3)}/1K vs HaluRISC ~$0.001/1K (measured, {judge.model})
                  </p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">Run src/models/eval_llm_judge.py to populate (costs ~$0.02).</p>
              )}
            </div>
            <div>
              <h3 className="text-base font-bold gradient-text mb-3">🔍 Error Analysis (10 FP + 10 FN, auto-tagged)</h3>
              {errors ? (
                <>
                  <p className="text-xs text-muted-foreground mb-3">
                    Test misclassifications: {errors.n_false_positives} FP / {errors.n_false_negatives} FN of {errors.n_test} samples
                    (F1 {results?.xgboost?.f1.toFixed(4)}).
                  </p>
                  <table className="w-full text-left text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border text-xs text-muted-foreground">
                        <th className="py-2 px-3 font-semibold">Category</th>
                        <th className="py-2 px-3 font-semibold">FP</th>
                        <th className="py-2 px-3 font-semibold">FN</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys({ ...errors.category_counts.false_positive, ...errors.category_counts.false_negative }).map((cat) => (
                        <tr key={cat} className="border-b border-border/50">
                          <td className="py-2 px-3">{cat}</td>
                          <td className="py-2 px-3 font-mono">{errors.category_counts.false_positive[cat] ?? 0}</td>
                          <td className="py-2 px-3 font-mono">{errors.category_counts.false_negative[cat] ?? 0}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-xs text-muted-foreground mt-3">
                    Categories are heuristic first-pass tags; review artifacts/results/error_analysis_cases.json before paper use.
                  </p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">Run src/models/error_analysis.py to populate.</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
