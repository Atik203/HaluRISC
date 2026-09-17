"use client";

import { createContext, useContext, type ReactNode } from "react";

/**
 * B7.5 Tier 1-3 — chat-wide auto-analysis configuration.
 * The chat page owns the evidence context, document index state, web-search
 * toggle and master toggle; AutoRiskCards consume them.
 */
export interface AutoAnalysisConfig {
  enabled: boolean;
  sessionContext: string | null;
  webEnabled?: boolean;
  hasDocuments?: boolean;
}

const AutoAnalysisContext = createContext<AutoAnalysisConfig>({
  enabled: true,
  sessionContext: null,
  webEnabled: false,
  hasDocuments: false,
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
