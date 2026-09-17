import type { ReactNode } from "react";
import { DashboardTabs } from "@/components/dashboard/dashboard-tabs";

export const dynamic = "force-dynamic";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="space-y-6">
      <div className="glass-panel p-5 md:p-6 rounded-2xl flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl md:text-2xl font-bold gradient-text">Experiment Dashboard</h1>
          <p className="text-xs text-muted-foreground mt-1">
            Every number is rendered from artifacts/results/ — no hardcoded metrics.
          </p>
        </div>
      </div>
      <DashboardTabs />
      {children}
    </div>
  );
}
