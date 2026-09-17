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

const AXIS = "var(--muted-foreground)";
const GRID = "rgba(148, 163, 184, 0.18)";
const TOOLTIP_STYLE = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 10,
  fontSize: "12px",
  color: "var(--popover-foreground)",
  boxShadow: "0 12px 32px rgba(0, 0, 0, 0.18)",
};
const LEGEND_STYLE = { fontSize: 11, color: AXIS };

export function GroupImportanceBars({
  data,
}: {
  data: Array<{ group: string; shap: number; ablation: number }>;
}) {
  return (
    <div className="w-full h-64" role="img" aria-label="Feature group mean SHAP versus ablation F1 delta">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="group" tick={{ fontSize: 11 }} stroke={AXIS} />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--secondary)", opacity: 0.35 }} />
          <Legend wrapperStyle={LEGEND_STYLE} />
          <Bar dataKey="shap" name="Group mean |SHAP|" fill={VIOLET} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="ablation" name="Ablation ΔF1" fill={CYAN} radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis
            dataKey="k"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 11 }}
            stroke={AXIS}
            label={{ value: "k neutralized features", fontSize: 11, position: "insideBottom", offset: -2, fill: AXIS }}
          />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Line type="monotone" dataKey="mean_score_delta" name="Mean Δscore" stroke={ROSE} strokeWidth={2} dot={{ r: 4, fill: ROSE }} isAnimationActive={false} />
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="perturbation" tick={{ fontSize: 10 }} stroke={AXIS} interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--secondary)", opacity: 0.35 }} />
          <Legend wrapperStyle={LEGEND_STYLE} />
          <Bar dataKey="mean_abs_score_delta" name="Mean |Δscore|" fill={INDIGO} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="top1_flip_rate" name="SHAP top-1 flip rate" fill={EMERALD} radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="subset" tick={{ fontSize: 10 }} stroke={AXIS} interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} domain={[0, 1]} />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--secondary)", opacity: 0.35 }} />
          <Legend wrapperStyle={LEGEND_STYLE} />
          <Bar dataKey="raw" name="Raw" fill={GRAY} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="platt" name="Platt" fill={VIOLET} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="isotonic" name="Isotonic" fill={CYAN} radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="subset" tick={{ fontSize: 10 }} stroke={AXIS} interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--secondary)", opacity: 0.35 }} />
          <Legend wrapperStyle={LEGEND_STYLE} />
          <Bar dataKey="f1" name="Zero-shot F1" fill={VIOLET} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="delta" name="ΔF1 vs in-domain" fill={ROSE} radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="feature" tick={{ fontSize: 10 }} stroke={AXIS} interval={0} />
          <YAxis tick={{ fontSize: 11 }} stroke={AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--secondary)", opacity: 0.35 }} />
          <Bar dataKey="mean" name="Mean |SHAP|" fill={INDIGO} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.feature} fill={INDIGO} />
            ))}
          </Bar>
          <Bar dataKey="hi" name="Upper bound" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="lo" name="Lower bound" fill="transparent" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
