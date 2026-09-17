import type { ReactNode } from "react";

/** Server-safe dashboard building blocks (no interactivity). */

export function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`glass-panel p-5 md:p-6 rounded-2xl space-y-4 ${className}`}>
      {title && (
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">{title}</h2>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
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
  const accents: Record<string, string> = {
    purple: "border-l-purple-500 bg-purple-500/20 text-violet-600 dark:text-purple-400",
    emerald: "border-l-emerald-500 bg-emerald-500/20 text-emerald-600 dark:text-emerald-400",
    blue: "border-l-blue-500 bg-blue-500/20 text-blue-600 dark:text-blue-400",
    amber: "border-l-amber-500 bg-amber-500/20 text-amber-600 dark:text-amber-400",
    rose: "border-l-rose-500 bg-rose-500/20 text-rose-600 dark:text-rose-400",
  };
  return (
    <div className={`glass-panel p-5 rounded-2xl border-l-4 ${accents[accent].split(" ")[0]} flex items-center gap-4`}>
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${accents[accent].split(" ").slice(1).join(" ")}`}>
        {icon}
      </div>
      <div>
        <div className="text-2xl font-extrabold">{value}</div>
        <div className="text-xs text-muted-foreground">{label}</div>
        {sub && <div className="text-[10px] text-muted-foreground/80 mt-0.5">{sub}</div>}
      </div>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="glass-panel p-8 rounded-2xl text-center">
      <p className="text-xs text-muted-foreground">{message}</p>
    </div>
  );
}

export function Figure({ src, alt, className = "w-full max-w-xl" }: { src: string; alt: string; className?: string }) {
  return <img src={src} alt={alt} className={`rounded-xl border border-border/60 bg-secondary/30 ${className}`} />;
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
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm border-collapse">
        <thead>
          <tr className="border-b border-border text-xs text-muted-foreground">
            {columns.map((c) => (
              <th key={c.key} className={`py-3 px-4 font-semibold ${c.align === "right" ? "text-right" : ""}`}>
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
                className={`border-b border-border/50 hover:bg-secondary/40 transition-colors ${
                  hl ? "bg-purple-500/10 font-semibold text-purple-700 dark:bg-purple-950/20 dark:text-purple-300" : ""
                }`}
              >
                {columns.map((c) => (
                  <td key={c.key} className={`py-2.5 px-4 ${c.align === "right" ? "text-right font-mono" : ""}`}>
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
  return <span className="font-mono">{v == null || Number.isNaN(v) ? "—" : v.toFixed(digits)}</span>;
}
