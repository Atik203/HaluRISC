"use client";

import React from "react";

interface RiskGaugeProps {
  score: number; // Calibrated probability between 0.0 and 1.0
  label?: string; // "low_risk" | "medium_risk" | "high_risk"
  thresholds?: { low?: number; medium?: number };
  latencyMs?: number;
}

const LABELS: Record<string, { color: string; text: string }> = {
  low_risk: { color: "#22c55e", text: "Low Risk" },
  medium_risk: { color: "#eab308", text: "Medium Risk" },
  high_risk: { color: "#ef4444", text: "High Risk" },
};

/** Polar point on the semicircle: f=0 → left, f=0.5 → top, f=1 → right. */
function arcPoint(fraction: number, radius: number): { x: number; y: number } {
  const angle = Math.PI * (1 - fraction);
  return { x: 100 + radius * Math.cos(angle), y: 100 - radius * Math.sin(angle) };
}

export function RiskGauge({ score, label, thresholds, latencyMs }: RiskGaugeProps) {
  const percentage = Math.min(100, Math.max(0, Math.round(score * 100)));
  const needleAngle = -90 + (percentage / 100) * 180;

  // API label is authoritative; fall back to percentage bands only if absent
  const fromLabel = label ? LABELS[label] : undefined;
  const statusColor = fromLabel?.color ?? (percentage >= 70 ? "#ef4444" : percentage >= 30 ? "#eab308" : "#22c55e");
  const statusText = fromLabel?.text ?? (percentage >= 70 ? "High Risk" : percentage >= 30 ? "Medium Risk" : "Low Risk");

  const ticks = [thresholds?.low, thresholds?.medium].filter(
    (t): t is number => typeof t === "number" && t > 0 && t < 1,
  );

  return (
    <div
      className="flex flex-col items-center justify-center py-2"
      role="img"
      aria-label={`Hallucination risk score ${percentage} percent, ${statusText}`}
    >
      {/* Semicircular arc gauge */}
      <div className="relative w-64 h-36 flex items-center justify-center">
        <svg viewBox="0 0 200 110" className="w-full h-full" aria-hidden>
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="50%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>

          {/* Background arc */}
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--gauge-track)" strokeWidth="16" strokeLinecap="round" />

          {/* Color gradient arc */}
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGradient)" strokeWidth="16" strokeLinecap="round" />

          {/* Decision-threshold notches */}
          {ticks.map((t) => {
            const inner = arcPoint(t, 68);
            const outer = arcPoint(t, 92);
            return (
              <line
                key={t}
                x1={inner.x}
                y1={inner.y}
                x2={outer.x}
                y2={outer.y}
                stroke="var(--card)"
                strokeWidth="2.5"
                strokeLinecap="round"
              />
            );
          })}

          {/* Center pivot */}
          <circle cx="100" cy="100" r="7" fill="var(--gauge-needle)" />

          {/* Needle */}
          <g transform={`rotate(${needleAngle}, 100, 100)`} className="transition-transform duration-1000 ease-out">
            <line x1="100" y1="96" x2="100" y2="30" stroke="var(--gauge-needle)" strokeWidth="4" strokeLinecap="round" />
          </g>
        </svg>
      </div>

      {/* Score readout */}
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="text-4xl font-extrabold tracking-tight font-mono tnum">{percentage}%</span>
        <span className="text-[11px] font-normal text-muted-foreground">calibrated probability</span>
      </div>

      {/* Status badge */}
      <div
        className="mt-3 rounded-full px-4 py-1 text-xs font-semibold uppercase tracking-wider"
        style={{
          backgroundColor: `${statusColor}1f`,
          color: statusColor,
          border: `1px solid ${statusColor}55`,
        }}
      >
        {statusText}
      </div>

      {latencyMs !== undefined && (
        <p className="mt-2 font-mono text-[10px] text-muted-foreground">latency {latencyMs.toFixed(0)} ms</p>
      )}
    </div>
  );
}
