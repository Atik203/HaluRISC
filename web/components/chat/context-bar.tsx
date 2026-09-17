"use client";

import { useState } from "react";
import {
  ChevronDown,
  FileText,
  Globe,
  Info,
  Loader2,
  MessagesSquare,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { MlStatus } from "@/components/ml-status";
import { CONTEXT_MAX } from "@/lib/analysis-input";

export interface ContextBarProps {
  evidence: string;
  onEvidenceChange: (v: string) => void;
  onClearEvidence: () => void;
  autoEnabled: boolean;
  onAutoChange: (v: boolean) => void;
  webEnabled: boolean;
  onWebChange: (v: boolean) => void;
  docCount: number;
  uploading: boolean;
  uploadError: string | null;
  onUpload: (files: FileList | File[]) => void;
  onClearDocuments: () => void;
}

export function ContextBar({
  evidence,
  onEvidenceChange,
  onClearEvidence,
  autoEnabled,
  onAutoChange,
  webEnabled,
  onWebChange,
  docCount,
  uploading,
  uploadError,
  onUpload,
  onClearDocuments,
}: ContextBarProps) {
  const [open, setOpen] = useState(false);
  const [dragging, setDragging] = useState(false);

  const hasEvidence = evidence.trim().length > 0;
  const grounding = hasEvidence
    ? "pasted text"
    : docCount > 0
      ? `${docCount} indexed passage${docCount === 1 ? "" : "s"}`
      : webEnabled
        ? "web search"
        : "conversation only";

  return (
    <section className="glass-panel rounded-2xl animate-fade">
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 shrink-0 rounded-xl surface-inset flex items-center justify-center">
            <MessagesSquare className="w-4.5 h-4.5 text-violet-600 dark:text-purple-400" aria-hidden />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-bold leading-tight">Chat with a grounded assistant</h1>
            <p className="text-[11px] text-muted-foreground truncate">
              Grounding: {grounding}
              {hasEvidence && ` · ${evidence.length.toLocaleString()} / ${CONTEXT_MAX.toLocaleString()} chars`}
            </p>
          </div>
        </div>
        <MlStatus />
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-border/40 px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="evidence-panel"
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-colors ${
              open
                ? "border-primary/40 bg-primary/10 text-foreground"
                : "border-border/70 bg-secondary/50 text-muted-foreground hover:text-foreground"
            }`}
          >
            <FileText className="w-3.5 h-3.5" aria-hidden />
            Evidence
            <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
          </button>
          {hasEvidence && (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-violet-500/30 bg-violet-500/10 px-2.5 py-1.5 text-[11px] font-mono text-violet-700 dark:text-violet-300">
              pasted text
              <button
                type="button"
                onClick={onClearEvidence}
                aria-label="Clear pasted evidence"
                className="text-violet-700/70 hover:text-violet-700 dark:text-violet-300/70 dark:hover:text-violet-200"
              >
                <X className="w-3 h-3" aria-hidden />
              </button>
            </span>
          )}
          {docCount > 0 && (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/70 bg-secondary/50 px-2.5 py-1.5 text-[11px] font-mono text-muted-foreground">
              {docCount} passages
              <button
                type="button"
                onClick={onClearDocuments}
                aria-label="Remove indexed documents"
                className="hover:text-rose-500"
              >
                <X className="w-3 h-3" aria-hidden />
              </button>
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1">
          <Switch
            checked={autoEnabled}
            onChange={onAutoChange}
            label="Auto risk check"
            icon={<ShieldCheck className="w-3.5 h-3.5" aria-hidden />}
            title="Score every assistant answer as soon as it finishes streaming"
          />
          <Switch
            checked={webEnabled}
            onChange={onWebChange}
            label="Web search"
            icon={<Globe className="w-3.5 h-3.5" aria-hidden />}
            title="Let claim checks search the web for evidence"
          />
        </div>
      </div>

      {/* Evidence panel */}
      {open && (
        <div
          id="evidence-panel"
          className="border-t border-border/40 p-4 space-y-3 animate-fade"
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (e.dataTransfer.files?.length) onUpload(e.dataTransfer.files);
          }}
        >
          <label htmlFor="chat-evidence" className="eyebrow block">
            Reference evidence
          </label>
          <textarea
            id="chat-evidence"
            value={evidence}
            onChange={(e) => onEvidenceChange(e.target.value.slice(0, CONTEXT_MAX))}
            placeholder="Paste a document, article, or reference material once. Every answer is then scored against it."
            rows={3}
            className={`w-full rounded-xl border bg-secondary/40 px-3 py-2.5 text-sm leading-relaxed transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
              dragging ? "border-primary/60 bg-primary/5" : "border-border"
            }`}
          />
          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-border bg-secondary/60 px-3 py-2 text-xs font-semibold transition-colors hover:bg-secondary">
              {uploading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden />
              ) : (
                <Upload className="w-3.5 h-3.5" aria-hidden />
              )}
              {uploading ? "Indexing…" : "Upload PDF, DOCX, TXT"}
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                className="sr-only"
                disabled={uploading}
                onChange={(e) => {
                  if (e.target.files) onUpload(e.target.files);
                  e.target.value = "";
                }}
              />
            </label>
            <span className="text-[11px] text-muted-foreground">or drop files here</span>
            <span className="text-[11px] text-muted-foreground sm:ml-auto">
              Priority: pasted text → documents → web → conversation
            </span>
          </div>
          {uploadError && (
            <p className="text-[11px] text-rose-600 dark:text-rose-400" role="alert">
              {uploadError}
            </p>
          )}
        </div>
      )}

      {/* Technical disclosure */}
      <details className="group border-t border-border/40">
        <summary className="flex cursor-pointer items-center gap-1.5 px-4 py-2 text-[11px] font-semibold text-muted-foreground hover:text-foreground">
          <Info className="w-3.5 h-3.5" aria-hidden />
          Technical details
        </summary>
        <div className="space-y-1.5 px-4 pb-3 text-[11px] leading-relaxed text-muted-foreground">
          <p>
            Answers stream through <code className="font-mono">/api/chat</code> (Vercel AI SDK) and need{" "}
            <code className="font-mono">OPENAI_API_KEY</code> in <code className="font-mono">web/.env.local</code>. Without a key,
            the risk tools still work in Analyze mode and the offline presenter demo.
          </p>
          <p>
            Risk checks call the local FastAPI service on port 8000 through the{" "}
            <code className="font-mono">/api/ml</code> rewrite.
          </p>
        </div>
      </details>
    </section>
  );
}
