"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AssistantRuntimeProvider, useAui, useAuiState } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import { AutoAnalysisProvider } from "@/components/assistant-ui/auto-analysis-context";
import { ContextBar } from "@/components/chat/context-bar";
import { ThreadSidebar } from "@/components/chat/thread-sidebar";
import { useChatThreadsRuntime } from "@/components/chat/use-chat-threads-runtime";
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

  useEffect(() => {
    fetch("/api/ml/index", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => setDocCount(s?.n_passages ?? 0))
      .catch(() => setDocCount(0));
  }, []);

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

  return (
    <div className="flex flex-col gap-3 h-[calc(100dvh-13rem)] min-h-[30rem]">
      <ContextBar
        evidence={evidence}
        onEvidenceChange={setEvidence}
        onClearEvidence={() => setEvidence("")}
        autoEnabled={autoEnabled}
        onAutoChange={setAutoEnabled}
        webEnabled={webEnabled}
        onWebChange={setWebEnabled}
        docCount={docCount}
        uploading={uploading}
        uploadError={uploadError}
        onUpload={uploadFiles}
        onClearDocuments={clearDocuments}
        onOpenHistory={() => setHistoryOpen(true)}
      />

      <AssistantRuntimeProvider runtime={runtime}>
        <AutoAnalysisProvider
          config={{
            enabled: autoEnabled,
            sessionContext: evidence.trim() || null,
            webEnabled,
            hasDocuments: docCount > 0,
          }}
        >
          <ThreadListSync />
          <div className="flex min-h-0 flex-1 gap-3">
            <ThreadSidebar className="hidden w-64 shrink-0 lg:flex" />
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
        </AutoAnalysisProvider>
      </AssistantRuntimeProvider>
    </div>
  );
}
