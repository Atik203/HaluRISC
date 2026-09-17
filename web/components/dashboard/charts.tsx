"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/** Client chart components (recharts) for the dashboard tabs. */

const VIOLET = "#8b5cf6";
const INDIGO = "#6366f1";
const CYAN = "#38bdf8";
const ROSE = "#f43f5e";
const EMERALD = "#10b981";
const GRAY = "#94a3b8";

export function GroupImportanceBars({
  data,
}: {
  data: Array<{ group: string; shap: number; ablation: number }>;
}) {
  return (
    <div className="w-full h-64" role="img" aria-label="Feature group mean SHAP versus ablation F1 delta">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="group" tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="shap" name="Group mean |SHAP|" fill={VIOLET} />
          <Bar dataKey="ablation" name="Ablation ΔF1" fill={CYAN} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function NeutralizationChart({
  data,
}: {
  data: Array<{ k: number; mean_score_delta: number }>;
}) {
  return (
    <div className="w-full h-56" role="img" aria-label="Mean score delta when top-k features are neutralized">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="k" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} stroke="#94a3b8" label={{ value: "k neutralized features", fontSize: 11, position: "insideBottom", offset: -2 }} />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Line type="monotone" dataKey="mean_score_delta" name="Mean Δscore" stroke={ROSE} strokeWidth={2} dot={{ r: 4, fill: ROSE }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PerturbationChart({
  data,
}: {
  data: Array<{ perturbation: string; mean_abs_score_delta: number; top1_flip_rate: number }>;
}) {
  return (
    <div className="w-full h-64" role="img" aria-label="Mean absolute score delta and top-1 SHAP flip rate per perturbation type">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="perturbation" tick={{ fontSize: 10 }} stroke="#94a3b8" interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="mean_abs_score_delta" name="Mean |Δscore|" fill={INDIGO} />
          <Bar dataKey="top1_flip_rate" name="SHAP top-1 flip rate" fill={EMERALD} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CalibrationBars({
  data,
}: {
  data: Array<{ subset: string; raw: number | null; platt: number | null; isotonic: number | null }>;
}) {
  return (
    <div className="w-full h-72" role="img" aria-label="ECE by subset and calibration method">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="subset" tick={{ fontSize: 10 }} stroke="#94a3b8" interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="raw" name="Raw" fill={GRAY} />
          <Bar dataKey="platt" name="Platt" fill={VIOLET} />
          <Bar dataKey="isotonic" name="Isotonic" fill={CYAN} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TransferBars({
  data,
}: {
  data: Array<{ subset: string; f1: number | null; delta: number | null }>;
}) {
  return (
    <div className="w-full h-64" role="img" aria-label="Zero-shot F1 and delta versus in-domain for each external subset">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="subset" tick={{ fontSize: 10 }} stroke="#94a3b8" interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="f1" name="Zero-shot F1" fill={VIOLET} />
          <Bar dataKey="delta" name="ΔF1 vs in-domain" fill={ROSE} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StabilityBars({
  data,
}: {
  data: Array<{ feature: string; mean: number; lo: number; hi: number }>;
}) {
  return (
    <div className="w-full h-64" role="img" aria-label="Mean absolute SHAP per feature with bootstrap bounds">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="feature" tick={{ fontSize: 10 }} stroke="#94a3b8" interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: "#1e1b2e", border: "1px solid rgba(148,163,184,0.3)", borderRadius: 8 }} />
          <Bar dataKey="mean" name="Mean |SHAP|" fill={INDIGO}>
            {data.map((d) => (
              <Cell key={d.feature} fill={INDIGO} />
            ))}
          </Bar>
          <Bar dataKey="hi" name="Upper bound" fill="transparent" />
          <Bar dataKey="lo" name="Lower bound" fill="transparent" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
