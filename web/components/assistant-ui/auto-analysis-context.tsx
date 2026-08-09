"use client";

import { createContext, useContext, type ReactNode } from "react";

/**
 * B7.5 Tier 1 — chat-wide auto-analysis configuration.
 * The chat page owns the evidence context + master toggle and shares them
 * with every AutoRiskCard rendered inside the Thread.
 */
export interface AutoAnalysisConfig {
  enabled: boolean;
  sessionContext: string | null;
}

const AutoAnalysisContext = createContext<AutoAnalysisConfig>({
  enabled: true,
  sessionContext: null,
});

export function AutoAnalysisProvider({
  config,
  children,
}: {
  config: AutoAnalysisConfig;
  children: ReactNode;
}) {
  return (
    <AutoAnalysisContext.Provider value={config}>{children}</AutoAnalysisContext.Provider>
  );
}

export function useAutoAnalysis(): AutoAnalysisConfig {
  return useContext(AutoAnalysisContext);
}
