"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, Globe, Loader2, Plus, ShieldCheck, Upload, X } from "lucide-react";
import { useChatControls } from "@/components/chat/chat-controls-context";
import { CONTEXT_MAX } from "@/lib/analysis-input";

function Chip({
  icon,
  label,
  onClear,
  tone = "default",
}: {
  icon: React.ReactNode;
  label: string;
  onClear: () => void;
  tone?: "default" | "violet" | "danger";
}) {
  const toneClass =
    tone === "violet"
      ? "border-violet-500/30 bg-violet-500/10 text-violet-700 dark:text-violet-300"
      : tone === "danger"
        ? "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
        : "border-border/70 bg-secondary/50 text-muted-foreground";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${toneClass}`}>
      {icon}
      {label}
      <button
        type="button"
        onClick={onClear}
        aria-label={`Clear ${label}`}
        className="opacity-70 transition-opacity hover:opacity-100"
      >
        <X className="h-3 w-3" aria-hidden />
      </button>
    </span>
  );
}

/** Evidence and attachment chips shown above the composer input. */
export function ComposerChips() {
  const { evidence, onClearEvidence, docCount, uploading, uploadError, onClearDocuments } = useChatControls();
  const hasEvidence = evidence.trim().length > 0;
  if (!hasEvidence && docCount === 0 && !uploading && !uploadError) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-1 pb-2">
      {hasEvidence && (
        <Chip
          icon={<FileText className="h-3 w-3" aria-hidden />}
          label={`Evidence · ${evidence.length.toLocaleString()} chars`}
          onClear={onClearEvidence}
          tone="violet"
        />
      )}
      {docCount > 0 && (
        <Chip
          icon={<FileText className="h-3 w-3" aria-hidden />}
          label={`${docCount} passage${docCount === 1 ? "" : "s"} indexed`}
          onClear={onClearDocuments}
        />
      )}
      {uploading && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-secondary/50 px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
          Indexing files…
        </span>
      )}
      {uploadError && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 px-2.5 py-1 text-[11px] font-medium text-rose-700 dark:text-rose-300" role="alert">
          {uploadError}
        </span>
      )}
    </div>
  );
}

/** Attachment menu, evidence popover, and the two runtime toggles. */
export function ComposerControls() {
  const controls = useChatControls();
  const [menuOpen, setMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const anyOpen = menuOpen || evidenceOpen;

  useEffect(() => {
    if (!anyOpen) return;
    const close = () => {
      setMenuOpen(false);
      setEvidenceOpen(false);
    };
    const onPointerDown = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) close();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [anyOpen]);

  const toggleChip = (active: boolean) =>
    `inline-flex h-8 items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-semibold transition-colors ${
      active
        ? "border-primary/50 bg-primary/15 text-foreground"
        : "border-border/70 bg-secondary/40 text-muted-foreground hover:text-foreground"
    }`;

  return (
    <div ref={rootRef} className="relative flex items-center gap-1">
      <button
        type="button"
        onClick={() => {
          setMenuOpen((v) => !v);
          setEvidenceOpen(false);
        }}
        aria-label="Add evidence or files"
        aria-expanded={menuOpen}
        title="Add evidence or files"
        className={`inline-flex h-9 w-9 items-center justify-center rounded-full border transition-colors ${
          menuOpen
            ? "border-primary/50 bg-primary/15 text-foreground"
            : "border-border/70 bg-secondary/40 text-muted-foreground hover:text-foreground"
        }`}
      >
        <Plus className="h-4 w-4" aria-hidden />
      </button>

      <button
        type="button"
        onClick={controls.onAutoChange.bind(null, !controls.autoEnabled)}
        aria-pressed={controls.autoEnabled}
        title="Score every assistant answer as soon as it finishes streaming"
        className={toggleChip(controls.autoEnabled)}
      >
        <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
        <span className="hidden sm:inline">Auto risk</span>
      </button>

      <button
        type="button"
        onClick={controls.onWebChange.bind(null, !controls.webEnabled)}
        aria-pressed={controls.webEnabled}
        title="Let claim checks search the web for evidence"
        className={toggleChip(controls.webEnabled)}
      >
        <Globe className="h-3.5 w-3.5" aria-hidden />
        <span className="hidden sm:inline">Web</span>
      </button>

      <input
        ref={fileRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md"
        className="sr-only"
        disabled={controls.uploading}
        onChange={(e) => {
          if (e.target.files) controls.onUpload(e.target.files);
          e.target.value = "";
        }}
      />

      {menuOpen && (
        <div className="absolute bottom-full left-0 z-40 mb-2 w-56 overflow-hidden rounded-2xl border border-border/70 bg-card/95 shadow-xl backdrop-blur-xl">
          <button
            type="button"
            onClick={() => {
              setMenuOpen(false);
              fileRef.current?.click();
            }}
            className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-xs font-semibold transition-colors hover:bg-secondary/60"
          >
            <Upload className="h-3.5 w-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
            Upload documents
            <span className="ml-auto text-[10px] font-normal text-muted-foreground">PDF, DOCX, TXT</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setMenuOpen(false);
              setEvidenceOpen(true);
            }}
            className="flex w-full items-center gap-2.5 border-t border-border/50 px-3.5 py-2.5 text-left text-xs font-semibold transition-colors hover:bg-secondary/60"
          >
            <FileText className="h-3.5 w-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
            Paste reference evidence
          </button>
        </div>
      )}

      {evidenceOpen && (
        <div className="absolute bottom-full left-0 z-40 mb-2 w-[min(24rem,80vw)] space-y-2.5 rounded-2xl border border-border/70 bg-card/95 p-3.5 shadow-xl backdrop-blur-xl">
          <div className="flex items-center justify-between gap-2">
            <span className="eyebrow">Reference evidence</span>
            <span className="font-mono text-[10px] text-muted-foreground">
              {controls.evidence.length.toLocaleString()} / {CONTEXT_MAX.toLocaleString()}
            </span>
          </div>
          <textarea
            value={controls.evidence}
            onChange={(e) => controls.onEvidenceChange(e.target.value.slice(0, CONTEXT_MAX))}
            placeholder="Paste a document, article, or reference material once. Every answer is then scored against it."
            rows={5}
            autoFocus
            className="w-full resize-y rounded-xl border border-border bg-secondary/40 px-3 py-2.5 text-xs leading-relaxed transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] text-muted-foreground">Priority: pasted text, then documents, web, conversation.</span>
            <button
              type="button"
              onClick={() => setEvidenceOpen(false)}
              className="rounded-lg bg-gradient-to-r from-violet-600 to-indigo-600 px-3 py-1.5 text-[11px] font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
