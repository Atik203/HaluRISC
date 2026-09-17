"use client";

import type { ReactNode } from "react";

export function Switch({
  checked,
  onChange,
  label,
  icon,
  title,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  icon?: ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      title={title}
      onClick={() => onChange(!checked)}
      className="inline-flex items-center gap-2 rounded-lg px-1.5 py-1 text-xs font-semibold text-muted-foreground transition-colors hover:text-foreground"
    >
      <span
        aria-hidden
        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-200 ${
          checked ? "border-primary/40 bg-primary" : "border-border bg-secondary"
        }`}
      >
        <span
          className={`absolute left-0.5 h-4 w-4 rounded-full shadow-sm transition-transform duration-200 ${
            checked ? "translate-x-4 bg-primary-foreground" : "translate-x-0 bg-foreground/60"
          }`}
        />
      </span>
      {icon}
      <span>{label}</span>
    </button>
  );
}
