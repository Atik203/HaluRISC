"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { AssistantRuntimeProvider, useAui, useAuiState } from "@assistant-ui/react";
import { Menu, PanelLeft } from "lucide-react";
import { Thread } from "@/components/assistant-ui/thread";
import { AutoAnalysisProvider } from "@/components/assistant-ui/auto-analysis-context";
import { ChatControlsProvider } from "@/components/chat/chat-controls-context";
import { ThreadSidebar } from "@/components/chat/thread-sidebar";
import { useChatThreadsRuntime } from "@/components/chat/use-chat-threads-runtime";
import { MlStatus } from "@/components/ml-status";
import { toast } from "@/components/ui/toast";

const SUGGESTIONS = [
  {
    title: "A case where the answer contradicts the given context",
    label: "Hallucinated answer",
    prompt:
      "Check this answer for hallucination risk. Question: 'What is the capital of France?' Context: 'France is a country in Europe. Its capital city is Paris.' Answer: 'The capital of France is Lyon, and it has been since 1800.'",
  },
  {
    title: "A case fully supported by the given context",
    label: "Grounded answer",
    prompt:
      "Check this answer for hallucination risk. Question: 'Who discovered penicillin?' Context: 'Penicillin was discovered by Alexander Fleming in 1928.' Answer: 'Penicillin was discovered by Alexander Fleming.'",
  },
  {
    title: "What the score means and how to read it",
    label: "How it works",
    prompt: "Explain how HaluRISC detects hallucinations and how to interpret the risk score.",
  },
  {
    title: "A case close to the low/medium decision boundary",
    label: "Borderline case",
    prompt:
      "Give me a borderline example where the risk score would be around 50% and explain why calibration matters.",
  },
];

/** localStorage-backed flag that stays hydration-safe (useSyncExternalStore). */
function useLocalStorageFlag(key: string, fallback: boolean) {
  const subscribe = useCallback((onChange: () => void) => {
    window.addEventListener("storage", onChange);
    window.addEventListener("halurisc-local", onChange);
    return () => {
      window.removeEventListener("storage", onChange);
      window.removeEventListener("halurisc-local", onChange);
    };
  }, []);
  const getSnapshot = useCallback(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw === null ? fallback : raw === "1";
    } catch {
      return fallback;
    }
  }, [key, fallback]);
  const getServerSnapshot = useCallback(() => fallback, [fallback]);
  const value = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const setValue = useCallback(
    (next: boolean) => {
      try {
        window.localStorage.setItem(key, next ? "1" : "0");
      } catch {
        /* storage unavailable */
      }
      window.dispatchEvent(new Event("halurisc-local"));
    },
    [key],
  );
  return [value, setValue] as const;
}

/** Refreshes sidebar titles/order once a run finishes (server-side auto-title). */
function ThreadListSync() {
  const aui = useAui();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const wasRunning = useRef(false);

  useEffect(() => {
    if (isRunning) {
      wasRunning.current = true;
      return;
    }
    if (!wasRunning.current) return;
    wasRunning.current = false;
    void aui.threads.reload().catch(() => {});
  }, [isRunning, aui]);

  return null;
}

function ChatTopBar({
  sidebarCollapsed,
  onToggleSidebar,
  onOpenMobile,
}: {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  onOpenMobile: () => void;
}) {
  const title = useAuiState(
    (s) => s.threads.threadItems.find((item) => item.id === s.threads.mainThreadId)?.title,
  );

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border/40 px-3">
      <button
        type="button"
        onClick={onOpenMobile}
        aria-label="Open chat history"
        className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground lg:hidden"
      >
        <Menu className="h-4 w-4" aria-hidden />
      </button>
      <button
        type="button"
        onClick={onToggleSidebar}
        aria-label={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        title={`${sidebarCollapsed ? "Show" : "Hide"} sidebar (Ctrl+B)`}
        className="hidden rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground lg:inline-flex"
      >
        <PanelLeft className="h-4 w-4" aria-hidden />
      </button>
      <span className="min-w-0 truncate text-xs font-semibold text-muted-foreground">
        {title?.trim() ? title : "New chat"}
      </span>
      <span className="ml-auto">
        <MlStatus />
      </span>
    </header>
  );
}

export default function ChatPage() {
  const [threadId, setThreadId] = useState<string | undefined>(undefined);

  // Deep links: /chat?thread=<id> opens a saved conversation.
  useEffect(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("thread");
    if (!fromUrl) return;
    queueMicrotask(() => setThreadId(fromUrl));
  }, []);

  const runtime = useChatThreadsRuntime({
    threadId,
    onThreadIdChange: (id) => {
      const url = new URL(window.location.href);
      if (id) url.searchParams.set("thread", id);
      else url.searchParams.delete("thread");
      window.history.replaceState(null, "", url.toString());
    },
  });

  const [evidence, setEvidence] = useState("");
  const [autoEnabled, setAutoEnabled] = useState(true);
  const [webEnabled, setWebEnabled] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useLocalStorageFlag("halurisc-chat-sidebar-collapsed", false);

  useEffect(() => {
    fetch("/api/ml/index", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => setDocCount(s?.n_passages ?? 0))
      .catch(() => setDocCount(0));
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "b") {
        event.preventDefault();
        setSidebarCollapsed(!sidebarCollapsed);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sidebarCollapsed, setSidebarCollapsed]);

  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files);
    if (list.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      const fd = new FormData();
      list.forEach((f) => fd.append("files", f));
      const res = await fetch("/api/ml/index", { method: "POST", body: fd });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail;
        throw new Error(detail || `upload failed (${res.status})`);
      }
      const body = (await res.json()) as { indexed?: Array<{ source: string; added: number }> };
      const added = (body.indexed ?? []).reduce((a, b) => a + b.added, 0);
      setDocCount((n) => n + added);
      toast(added > 0 ? `${added} passage${added === 1 ? "" : "s"} indexed` : "Files processed, no new passages found");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed.";
      setUploadError(message);
      toast(message, "error");
    } finally {
      setUploading(false);
    }
  }, []);

  const clearDocuments = useCallback(async () => {
    try {
      await fetch("/api/ml/index", { method: "DELETE" });
      setDocCount(0);
    } catch {
      /* ignore */
    }
  }, []);

  const controls = {
    autoEnabled,
    onAutoChange: setAutoEnabled,
    webEnabled,
    onWebChange: setWebEnabled,
    evidence,
    onEvidenceChange: setEvidence,
    onClearEvidence: () => setEvidence(""),
    docCount,
    uploading,
    uploadError,
    onUpload: uploadFiles,
    onClearDocuments: clearDocuments,
  };

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatControlsProvider value={controls}>
        <AutoAnalysisProvider
          config={{
            enabled: autoEnabled,
            sessionContext: evidence.trim() || null,
            webEnabled,
            hasDocuments: docCount > 0,
          }}
        >
          <div className="flex min-h-0 flex-1">
            <div
              className={`hidden shrink-0 transition-[width] duration-200 lg:flex ${
                sidebarCollapsed ? "w-0 overflow-hidden" : "w-72"
              }`}
            >
              <ThreadSidebar
                flush
                className="w-72"
                onToggleCollapse={() => setSidebarCollapsed(true)}
                onNavigate={() => setHistoryOpen(false)}
              />
            </div>

            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <ChatTopBar
                sidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                onOpenMobile={() => setHistoryOpen(true)}
              />
              <ThreadListSync />
              <Thread starters={SUGGESTIONS} />
            </div>

      {/* Mobile history drawer */}
      {historyOpen && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Chat history">
          <button
            type="button"
            aria-label="Close chat history"
            onClick={() => setHistoryOpen(false)}
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          />
          <div className="absolute left-0 top-0 h-full w-72 max-w-[85vw] p-3">
            <ThreadSidebar className="h-full" onNavigate={() => setHistoryOpen(false)} />
          </div>
        </div>
      )}
          </div>
        </AutoAnalysisProvider>
      </ChatControlsProvider>
    </AssistantRuntimeProvider>
  );
}
