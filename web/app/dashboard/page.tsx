"use client";

import React from "react";
import { LayoutDashboard, Award, Zap, DollarSign, CheckCircle2, TrendingUp } from "lucide-react";

export default function DashboardPage() {
  const models = [
    { name: "Heuristic Baseline (Overlap)", precision: "0.939", recall: "0.947", f1: "0.943", auroc: "0.915", pr_auc: "0.812", mcc: "0.885" },
    { name: "Logistic Regression", precision: "0.979", recall: "0.961", f1: "0.970", auroc: "0.994", pr_auc: "0.993", mcc: "0.940" },
    { name: "Random Forest", precision: "0.988", recall: "0.984", f1: "0.986", auroc: "0.998", pr_auc: "0.998", mcc: "0.972" },
    { name: "XGBoost (Calibrated - Ours)", precision: "0.990", recall: "0.982", f1: "0.986", auroc: "0.997", pr_auc: "0.998", mcc: "0.972", highlight: true },
    { name: "GPT 5.6 Luna Judge (Baseline)", precision: "0.945", recall: "0.950", f1: "0.947", auroc: "0.952", pr_auc: "0.940", mcc: "0.890" },
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
            <LayoutDashboard className="w-6 h-6 text-purple-400" /> 📈 Experiment Dashboard & Benchmarks
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Empirical evaluation results on 3,000 holdout test samples (HaluEval QA dataset)
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-purple-500 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <div className="text-2xl font-extrabold">0.986</div>
            <div className="text-xs text-muted-foreground">Test F1-Score (XGBoost)</div>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-emerald-500 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <div className="text-2xl font-extrabold">0.997</div>
            <div className="text-xs text-muted-foreground">AUROC Score</div>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-blue-500 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-blue-500/20 text-blue-400 flex items-center justify-center">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-2xl font-extrabold">~12 ms</div>
            <div className="text-xs text-muted-foreground">CPU Latency / Sample</div>
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
          Model Benchmarks Comparison (Test Split N=3,000)
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-border text-xs text-muted-foreground">
                <th className="py-3 px-4 font-semibold">Model Architecture</th>
                <th className="py-3 px-4 font-semibold">Precision</th>
                <th className="py-3 px-4 font-semibold">Recall</th>
                <th className="py-3 px-4 font-semibold">F1-Score</th>
                <th className="py-3 px-4 font-semibold">AUROC</th>
                <th className="py-3 px-4 font-semibold">PR-AUC</th>
                <th className="py-3 px-4 font-semibold">MCC</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, idx) => (
                <tr
                  key={idx}
                  className={`border-b border-border/50 hover:bg-secondary/40 transition-colors ${
                    m.highlight ? "bg-purple-950/20 font-semibold text-purple-300" : ""
                  }`}
                >
                  <td className="py-3 px-4 flex items-center gap-2">
                    {m.highlight && <CheckCircle2 className="w-4 h-4 text-purple-400" />}
                    <span>{m.name}</span>
                  </td>
                  <td className="py-3 px-4 font-mono">{m.precision}</td>
                  <td className="py-3 px-4 font-mono">{m.recall}</td>
                  <td className="py-3 px-4 font-mono">{m.f1}</td>
                  <td className="py-3 px-4 font-mono">{m.auroc}</td>
                  <td className="py-3 px-4 font-mono">{m.pr_auc}</td>
                  <td className="py-3 px-4 font-mono">{m.mcc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cost & Efficiency Advocacy Card */}
      <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-base font-bold gradient-text mb-2">⚡ Efficiency & Cost Advantage</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            HaluRISC extracts lightweight linguistic, entity, and numeric features in CPU milliseconds.
            Evaluating 10,000 responses with an LLM judge costs <strong>~$1.10</strong> via API, whereas HaluRISC runs locally at near-zero incremental cost.
          </p>
        </div>
        <div className="bg-secondary/40 p-4 rounded-xl border border-border/60 flex flex-col justify-center space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">GPT 5.6 Luna Judge (10K queries):</span>
            <span className="font-mono text-red-400">$1.10</span>
          </div>
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-muted-foreground">HaluRISC XGBoost (10K queries):</span>
            <span className="font-mono text-emerald-400">$0.00 (Local CPU)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
