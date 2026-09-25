import { ArrowRight } from "lucide-react";

/* Projector-safe palette (same family as the FYDP deck reference). */
export const NEAR_BLACK = "#0a0a0a";
export const DEEP_INK = "#101828";
export const SLATE = "#475569";
export const ACCENT = "#1e40af";
export const TEAL = "#0f766e";
export const AMBER = "#b45309";
export const ROSE = "#b91c1c";

export const FOOTER_TEXT = "CSE 4889 - Machine Learning · Section E · Team Phantom Devs";

/* ------------------------------------------------------------------ */
/* Frame: header + body + footer, with subtle corner accents           */
/* ------------------------------------------------------------------ */

export function SlideFrame({
  badge,
  badgeBg,
  badgeColor,
  title,
  subtitle,
  accent = ACCENT,
  index,
  total,
  footerLeft,
  children,
}: {
  badge: string;
  badgeBg: string;
  badgeColor: string;
  title: string;
  subtitle?: string;
  accent?: string;
  index: number;
  total: number;
  footerLeft?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-white px-[4.2cqw] pb-[1.4cqh] pt-[2.4cqh]">
      <div
        className="pointer-events-none absolute right-[-10cqw] top-[-16cqh] h-[46cqh] w-[46cqh] rounded-full"
        style={{ background: accent, opacity: 0.06 }}
      />
      <div
        className="pointer-events-none absolute bottom-[-18cqh] left-[-8cqw] h-[36cqh] w-[36cqh] rounded-full"
        style={{ background: accent, opacity: 0.04 }}
      />
      <SlideHeader
        badge={badge}
        badgeBg={badgeBg}
        badgeColor={badgeColor}
        title={title}
        subtitle={subtitle}
      />
      <div className="relative flex min-h-0 flex-1 flex-col">{children}</div>
      <div
        className="relative mt-[1.1cqh] flex items-center justify-between border-t-2 pt-[0.8cqh] text-[1.7cqh] font-bold"
        style={{ borderColor: "#e2e8f0", color: SLATE }}
      >
        <span>{footerLeft ?? FOOTER_TEXT}</span>
        <span className="tnum">
          {index + 1} / {total}
        </span>
      </div>
    </div>
  );
}

export function Card({
  icon,
  title,
  color,
  children,
  className = "",
  fill = "#ffffff",
  center = true,
}: {
  icon: React.ReactNode;
  title: string;
  color: string;
  children: React.ReactNode;
  className?: string;
  fill?: string;
  center?: boolean;
}) {
  return (
    <div
      className={`flex h-full min-h-0 flex-col rounded-xl border-2 px-[2.2cqw] py-[1.7cqh] ${className}`}
      style={{ borderColor: color, background: fill }}
    >
      <div className="mb-[1.1cqh] flex items-center gap-[1cqw]">
        <span
          className="flex flex-shrink-0 items-center justify-center rounded-lg"
          style={{ width: "4.2cqh", height: "4.2cqh", background: color }}
        >
          {icon}
        </span>
        <span
          className="text-[2.7cqh] font-extrabold uppercase tracking-wide"
          style={{ color }}
        >
          {title}
        </span>
      </div>
      <div className={`min-h-0 flex-1 ${center ? "flex flex-col justify-center" : ""}`}>
        {children}
      </div>
    </div>
  );
}

export function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="mb-[0.85cqh] flex items-start gap-[0.9cqw] last:mb-0">
      <span
        className="mt-[1.15cqh] flex-shrink-0 rounded-full"
        style={{ width: "1.15cqh", height: "1.15cqh", background: DEEP_INK }}
      />
      <span className="text-[2.45cqh] font-medium leading-snug" style={{ color: NEAR_BLACK }}>
        {children}
      </span>
    </li>
  );
}

export function SlideHeader({
  badge,
  badgeBg,
  badgeColor,
  title,
  subtitle,
}: {
  badge: string;
  badgeBg: string;
  badgeColor: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="relative mb-[1.5cqh]">
      <div
        className="inline-block rounded px-[1.6cqw] py-[0.45cqh] text-[1.9cqh] font-bold uppercase tracking-wider"
        style={{ background: badgeBg, color: badgeColor }}
      >
        {badge}
      </div>
      <h1
        className="mt-[1cqh] text-[3.9cqh] font-extrabold leading-tight"
        style={{ color: NEAR_BLACK }}
      >
        {title}
      </h1>
      {subtitle && (
        <div
          className="mt-[0.6cqh] text-[2.2cqh] font-semibold"
          style={{ color: DEEP_INK }}
        >
          {subtitle}
        </div>
      )}
    </div>
  );
}

export function Kpi({
  value,
  label,
  color,
  sub,
}: {
  value: string;
  label: string;
  color: string;
  sub?: string;
}) {
  return (
    <div className="text-center">
      <div className="tnum text-[6.6cqh] font-extrabold leading-none" style={{ color }}>
        {value}
      </div>
      <div className="mt-[0.7cqh] text-[2.25cqh] font-bold leading-snug" style={{ color: NEAR_BLACK }}>
        {label}
      </div>
      {sub && (
        <div className="mt-[0.35cqh] text-[1.85cqh] font-semibold" style={{ color: SLATE }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function KpiStrip({
  items,
  cols,
}: {
  items: Array<{ value: string; label: string; color: string; sub?: string }>;
  cols?: number;
}) {
  return (
    <div
      className="grid gap-[1.4cqw] rounded-xl px-[2cqw] py-[1.5cqh]"
      style={{
        background: "#f1f5f9",
        gridTemplateColumns: `repeat(${cols ?? items.length}, minmax(0, 1fr))`,
      }}
    >
      {items.map((item) => (
        <Kpi key={item.label} {...item} />
      ))}
    </div>
  );
}

export function PipelineStrip({
  steps,
  color = AMBER,
  bg = "#fffbeb",
}: {
  steps: string[];
  color?: string;
  bg?: string;
}) {
  return (
    <div className="flex items-stretch gap-[0.6cqw]">
      {steps.map((step, i, arr) => (
        <div key={step} className="flex min-w-0 flex-1 items-center">
          <div
            className="flex h-full flex-1 items-center justify-center rounded-lg border-2 px-[1.2cqw] py-[1.05cqh] text-center text-[1.95cqh] font-bold"
            style={{ borderColor: color, background: bg, color: NEAR_BLACK }}
          >
            {step}
          </div>
          {i < arr.length - 1 && (
            <ArrowRight size="2.5cqh" style={{ color }} className="mx-[0.3cqw] flex-shrink-0" />
          )}
        </div>
      ))}
    </div>
  );
}

export function Chip({
  children,
  color,
  bg,
}: {
  children: React.ReactNode;
  color: string;
  bg: string;
}) {
  return (
    <span
      className="inline-block rounded-lg px-[1.3cqw] py-[0.5cqh] text-[2.05cqh] font-bold"
      style={{ background: bg, color }}
    >
      {children}
    </span>
  );
}

export function SlideTable({
  head,
  rows,
  color = ACCENT,
  headerBg = "#e0e7ff",
  highlightRow,
  widths,
  compact = false,
}: {
  head: string[];
  rows: Array<Array<React.ReactNode>>;
  color?: string;
  headerBg?: string;
  highlightRow?: number;
  widths?: string[];
  compact?: boolean;
}) {
  return (
    <div className="overflow-hidden rounded-xl border-2" style={{ borderColor: color }}>
      <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
        <thead>
          <tr style={{ background: headerBg }}>
            {head.map((cell, i) => (
              <th
                key={cell}
                className={`px-[1.1cqw] font-extrabold ${compact ? "py-[0.8cqh] text-[2cqh]" : "py-[1cqh] text-[2.15cqh]"}`}
                style={{ color: NEAR_BLACK, width: widths?.[i] }}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              style={{
                background: ri === highlightRow ? "#eef2ff" : ri % 2 ? "#f8fafc" : "#ffffff",
              }}
            >
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className={`tnum px-[1.1cqw] font-semibold ${compact ? "py-[0.7cqh] text-[2cqh]" : "py-[0.85cqh] text-[2.05cqh]"} ${
                    ci === 0 ? "text-left" : "text-center"
                  }`}
                  style={{ color: NEAR_BLACK, borderTop: "1px solid #e2e8f0" }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Figure({
  src,
  alt,
  caption,
  height = "42cqh",
}: {
  src: string;
  alt: string;
  caption?: string;
  height?: string;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col items-center justify-center">
      <div
        className="flex min-h-0 items-center justify-center rounded-lg border border-slate-200 bg-white p-[0.8cqh]"
        style={{ maxHeight: height }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} className="max-h-full max-w-full object-contain" />
      </div>
      {caption && (
        <div className="mt-[0.7cqh] text-center text-[1.9cqh] font-semibold" style={{ color: SLATE }}>
          {caption}
        </div>
      )}
    </div>
  );
}

/** Framed app screenshot that crops to the informative region. */
export function SlideShot({
  src,
  alt,
  caption,
  color = ACCENT,
  position = "center",
  contain = false,
}: {
  src: string;
  alt: string;
  caption?: string;
  color?: string;
  position?: string;
  contain?: boolean;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        className="min-h-0 flex-1 overflow-hidden rounded-xl border-2 shadow-lg"
        style={{ borderColor: color, background: "#0b1020" }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          className={`h-full w-full ${contain ? "object-contain" : "object-cover"}`}
          style={{ objectPosition: position }}
        />
      </div>
      {caption && (
        <div className="mt-[0.6cqh] text-center text-[1.95cqh] font-bold" style={{ color }}>
          {caption}
        </div>
      )}
    </div>
  );
}

/** Horizontal proportion bar (splits, feature composition, verdict mix). */
export function StackedBar({
  segments,
  height = "4.4cqh",
}: {
  segments: Array<{ label: string; value: number; color: string }>;
  height?: string;
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  return (
    <div
      className="flex w-full overflow-hidden rounded-lg border-2 border-slate-200"
      style={{ height }}
    >
      {segments.map((segment) => (
        <div
          key={segment.label}
          className="tnum flex items-center justify-center text-[1.95cqh] font-extrabold text-white"
          style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }}
        >
          {segment.label}
        </div>
      ))}
    </div>
  );
}

export function fmt(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
}

export function ms(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? "—" : `${v.toFixed(1)} ms`;
}
