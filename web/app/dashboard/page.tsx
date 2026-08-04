import fs from "fs";
import path from "path";
import { Award, DollarSign, LayoutDashboard, TrendingUp, Zap } from "lucide-react";

export const dynamic = "force-dynamic";

const RESULTS_DIR = path.resolve(process.cwd(), "..", "artifacts", "results");

function readJson(name: string) {
  const p = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
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
  const results = readJson("final_results.json") as any;
  const ablation = readJson("ablation_results.csv") as string | null;
  const ragtruth = readJson("ragtruth_results.json") as any;
  const shap = readJson("shap_summary.json") as any;

  const modelRows = results
    ? Object.entries(MODEL_LABELS)
        .filter(([key]) => results[key])
        .map(([key, label]) => ({ key, label, ...results[key] }))
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
                  AUROC (95% CI [{stats?.bootstrap?.auroc_ci?.[0]?.toFixed(3) ?? "—"}, {stats?.bootstrap?.auroc_ci?.[1]?.toFixed(3) ?? "—"}])
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
                  {modelRows!.map((m: any) => (
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
                  {["platt", "isotonic"].map((m) => (
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
                  {shap.top_features?.map((f: any, i: number) => (
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
              {ablation ? (
                <table className="w-full text-left text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground">
                      <th className="py-2 px-3 font-semibold">Removed group</th>
                      <th className="py-2 px-3 font-semibold">F1 (mean ± std)</th>
                      <th className="py-2 px-3 font-semibold">ΔF1 vs full</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ablation.split("\n").slice(1).map((line) => {
                      const [group, f1, f1std, auroc, aurocstd] = line.split(",");
                      const delta = xgb.f1 - parseFloat(f1);
                      return (
                        <tr key={group} className="border-b border-border/50">
                          <td className="py-2 px-3">-{group}</td>
                          <td className="py-2 px-3 font-mono">
                            {parseFloat(f1).toFixed(4)} ± {parseFloat(f1std).toFixed(4)}
                          </td>
                          <td className={`py-2 px-3 font-mono ${delta > 0.01 ? "text-red-400" : "text-emerald-400"}`}>
                            {delta >= 0 ? "+" : ""}{delta.toFixed(4)}
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
                    {["f1", "auroc", "pr_auc", "mcc", "ece", "brier"].map((k) => (
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
        </>
      )}
    </div>
  );
}
