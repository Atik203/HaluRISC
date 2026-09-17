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
    <nav aria-label="Dashboard sections" className="flex flex-wrap gap-1.5">
      {TABS.map((tab) => {
        const Icon = tab.icon;
        const active = pathname === tab.href || pathname.startsWith(`${tab.href}/`);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            aria-current={active ? "page" : undefined}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border transition-all duration-200 ${
              active
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "text-muted-foreground border-border/60 hover:text-foreground hover:bg-secondary/60"
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
