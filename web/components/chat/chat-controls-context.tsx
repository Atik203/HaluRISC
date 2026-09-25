"use client";

import { createContext, useContext } from "react";

export interface ChatControls {
  autoEnabled: boolean;
  onAutoChange: (v: boolean) => void;
  webEnabled: boolean;
  onWebChange: (v: boolean) => void;
  evidence: string;
  onEvidenceChange: (v: string) => void;
  onClearEvidence: () => void;
  docCount: number;
  uploading: boolean;
  uploadError: string | null;
  onUpload: (files: FileList | File[]) => void;
  onClearDocuments: () => void;
}

const ChatControlsContext = createContext<ChatControls | null>(null);

export function ChatControlsProvider({
  value,
  children,
}: {
  value: ChatControls;
  children: React.ReactNode;
}) {
  return <ChatControlsContext.Provider value={value}>{children}</ChatControlsContext.Provider>;
}

export function useChatControls() {
  const ctx = useContext(ChatControlsContext);
  if (!ctx) throw new Error("useChatControls must be used inside ChatControlsProvider");
  return ctx;
}
