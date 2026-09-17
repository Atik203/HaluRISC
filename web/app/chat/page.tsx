"use client";

import { useCallback, useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { AutoAnalysisProvider } from "@/components/assistant-ui/auto-analysis-context";
import { ContextBar } from "@/components/chat/context-bar";

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

export default function ChatPage() {
  const runtime = useChatRuntime();
  const [evidence, setEvidence] = useState("");
  const [autoEnabled, setAutoEnabled] = useState(true);
  const [webEnabled, setWebEnabled] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

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
      setDocCount((n) => n + (body.indexed ?? []).reduce((a, b) => a + b.added, 0));
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
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
    <div className="flex flex-col h-[calc(100dvh-13rem)] min-h-[30rem] gap-3">
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
          <Thread starters={SUGGESTIONS} />
        </AutoAnalysisProvider>
      </AssistantRuntimeProvider>
    </div>
  );
}
