import type { ReactNode } from "react";
import { FigureLightbox } from "@/components/ui/figure-lightbox";

/** Server-safe dashboard building blocks (no interactivity). */

export function Panel({
  id,
  title,
  subtitle,
  children,
  className = "",
}: {
  id?: string;
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={`glass-panel p-5 md:p-6 rounded-2xl space-y-4 ${className}`}>
      {title && (
        <div className="border-b border-border/50 pb-3">
          <h2 className="text-sm font-bold tracking-tight">{title}</h2>
          {subtitle && <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}

export function KpiCard({
  icon,
  label,
  value,
  accent = "purple",
  sub,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  accent?: "purple" | "emerald" | "blue" | "amber" | "rose";
  sub?: string;
}) {
  const accents: Record<string, { border: string; chip: string }> = {
    purple: {
      border: "border-l-violet-500",
      chip: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
    },
    emerald: {
      border: "border-l-emerald-500",
      chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    },
    blue: {
      border: "border-l-sky-500",
      chip: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
    },
    amber: {
      border: "border-l-amber-500",
      chip: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
    },
    rose: {
      border: "border-l-rose-500",
      chip: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
    },
  };
  const a = accents[accent];
  return (
    <div className={`glass-panel rounded-2xl border-l-4 p-5 ${a.border}`}>
      <div className="flex items-center justify-between gap-3">
        <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${a.chip}`}>{icon}</div>
        {sub && <span className="text-[10px] font-mono text-muted-foreground">{sub}</span>}
      </div>
      <div className="mt-3 font-mono text-2xl md:text-[1.75rem] font-extrabold tracking-tight tnum">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="glass-panel rounded-2xl p-8 text-center">
      <p className="mx-auto max-w-md text-xs leading-relaxed text-muted-foreground">{message}</p>
    </div>
  );
}

export function Figure({
  src,
  alt,
  className = "w-full max-w-xl",
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  return <FigureLightbox src={src} alt={alt} className={className} />;
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  rows,
  rowKey,
  highlight,
}: {
  columns: Array<{ key: string; label: string; align?: "left" | "right" }>;
  rows: T[];
  rowKey: (row: T) => string;
  highlight?: (row: T) => boolean;
}) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No data.</p>;
  return (
    <div className="overflow-x-auto rounded-xl border border-border/50">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/60">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={`whitespace-nowrap px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground ${
                  c.align === "right" ? "text-right" : ""
                }`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const hl = highlight?.(row);
            return (
              <tr
                key={rowKey(row)}
                className={`border-b border-border/40 transition-colors last:border-0 hover:bg-secondary/40 ${
                  hl ? "bg-violet-500/[0.07] font-semibold" : ""
                }`}
              >
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={`px-4 py-2.5 text-[13px] ${c.align === "right" ? "text-right font-mono tnum" : ""}`}
                  >
                    {String(row[c.key] ?? "—")}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Mono({ v, digits = 4 }: { v: number | null | undefined; digits?: number }) {
  return <span className="font-mono tnum">{v == null || Number.isNaN(v) ? "—" : v.toFixed(digits)}</span>;
}
