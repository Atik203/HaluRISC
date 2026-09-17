"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Gauge,
  Layers,
  ShieldCheck,
} from "lucide-react";

const TABS = [
  { href: "/dashboard/overview", label: "Overview", icon: Layers },
  { href: "/dashboard/robustness", label: "Robustness", icon: ShieldCheck },
  { href: "/dashboard/calibration", label: "Calibration", icon: Gauge },
  { href: "/dashboard/explainability", label: "Explainability", icon: BarChart3 },
  { href: "/dashboard/failures", label: "Failures", icon: AlertTriangle },
  { href: "/dashboard/efficiency", label: "Efficiency", icon: Activity },
];

export function DashboardTabs() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Dashboard sections"
      className="-mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1"
    >
      {TABS.map((tab) => {
        const Icon = tab.icon;
        const active = pathname === tab.href || pathname.startsWith(`${tab.href}/`);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            aria-current={active ? "page" : undefined}
            className={`inline-flex shrink-0 items-center gap-2 rounded-xl border px-4 py-2 text-xs font-semibold transition-all duration-200 ${
              active
                ? "border-transparent bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md shadow-violet-600/20"
                : "border-border/60 text-muted-foreground hover:border-border hover:bg-secondary/60 hover:text-foreground"
            }`}
          >
            <Icon className="w-4 h-4" aria-hidden />
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
