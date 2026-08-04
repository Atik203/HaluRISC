"use client";

import React from "react";

interface RiskGaugeProps {
  score: number; // Calibrated probability between 0.0 and 1.0
  label?: string; // "low_risk" | "medium_risk" | "high_risk"
  latencyMs?: number;
}

export function RiskGauge({ score, label, latencyMs }: RiskGaugeProps) {
  const percentage = Math.min(100, Math.max(0, Math.round(score * 100)));
  
  // Calculate angle for needle: 0% -> -90 deg (left), 100% -> +90 deg (right)
  const needleAngle = -90 + (percentage / 100) * 180;

  let statusColor = "#22c55e"; // Green
  let statusText = "Low Risk";

  if (percentage >= 70) {
    statusColor = "#ef4444"; // Red
    statusText = "High Risk";
  } else if (percentage >= 30) {
    statusColor = "#eab308"; // Yellow
    statusText = "Medium Risk";
  }

  return (
    <div className="flex flex-col items-center justify-center p-6 rounded-2xl glass-panel relative overflow-hidden shadow-2xl">
      <div className="text-xs uppercase tracking-widest text-muted-foreground font-semibold mb-2">
        Hallucination Risk Score
      </div>

      {/* SVG Semicircular Arc Gauge */}
      <div className="relative w-64 h-36 flex items-center justify-center">
        <svg viewBox="0 0 200 110" className="w-full h-full">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="50%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>

          {/* Background Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth="16"
            strokeLinecap="round"
          />

          {/* Color Gradient Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="16"
            strokeLinecap="round"
          />

          {/* Center Pivot */}
          <circle cx="100" cy="100" r="8" fill="#ffffff" />

          {/* Animated Needle */}
          <g transform={`rotate(${needleAngle}, 100, 100)`} className="transition-transform duration-1000 ease-out">
            <line
              x1="100"
              y1="100"
              x2="100"
              y2="30"
              stroke="#ffffff"
              strokeWidth="4"
              strokeLinecap="round"
            />
          </g>
        </svg>
      </div>

      {/* Score readout */}
      <div className="text-4xl font-extrabold tracking-tight mt-2 flex items-baseline gap-1">
        <span>{percentage}%</span>
        <span className="text-xs font-normal text-muted-foreground">probability</span>
      </div>

      {/* Status Badge */}
      <div
        className="mt-3 px-4 py-1 rounded-full text-xs font-semibold uppercase tracking-wider shadow-lg"
        style={{
          backgroundColor: `${statusColor}20`,
          color: statusColor,
          border: `1px solid ${statusColor}40`,
        }}
      >
        {statusText}
      </div>

      {latencyMs !== undefined && (
        <div className="text-[10px] text-muted-foreground mt-3">
          Inference latency: {latencyMs} ms
        </div>
      )}
    </div>
  );
}
