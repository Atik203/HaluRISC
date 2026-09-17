"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowRight, CheckCircle2, RefreshCw, XCircle } from "lucide-react";

type CheckState = "loading" | "ok" | "warn" | "fail";

interface Check {
  key: string;
  label: string;
  state: CheckState;
  detail: string;
  hint?: string;
}

function StateIcon({ state }: { state: CheckState }) {
  if (state === "ok") return <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" aria-hidden />;
  if (state === "warn") return <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" aria-hidden />;
  if (state === "fail") return <XCircle className="h-4 w-4 text-rose-600 dark:text-rose-400" aria-hidden />;
  return <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" aria-hidden />;
}

/**
 * Booth preflight: verifies the live-demo dependencies before a presenter starts.
 * Artifact status comes from the server; the rest is probed live.
 */
export function DemoReadiness({
  artifactsReady,
  modelsReady,
  resultsReady,
}: {
  artifactsReady: boolean;
  modelsReady: boolean;
  resultsReady: boolean;
}) {
  const [checks, setChecks] = useState<Check[]>([]);
  const [busy, setBusy] = useState(true);

  const probe = useCallback(async () => {
    await Promise.resolve(); // defer setState out of the effect (no cascading renders)
    setBusy(true);
    const next: Check[] = [
      {
        key: "artifacts",
        label: "Frozen artifacts",
        state: artifactsReady && modelsReady && resultsReady ? "ok" : "fail",
        detail:
          artifactsReady && modelsReady && resultsReady
            ? "manifest, model files, and result tables present"
            : "some files are missing",
        hint: "Run the pipeline or unzip the Colab artifact bundle at the repo root.",
      },
    ];

    try {
      const res = await fetch("/api/ml/health", { cache: "no-store" });
      const data = res.ok
        ? ((await res.json()) as { status?: string; model?: string; artifacts_loaded?: boolean })
        : null;
      if (res.ok && data?.status === "ok" && data.artifacts_loaded) {
        next.push({ key: "backend", label: "ML backend", state: "ok", detail: data.model ?? "ready" });
      } else {
        next.push({
          key: "backend",
          label: "ML backend",
          state: "warn",
          detail: "starting or degraded",
          hint: "uvicorn src.api.main:app --port 8000 (first load takes 1-3 min)",
        });
      }
    } catch {
      next.push({
        key: "backend",
        label: "ML backend",
        state: "fail",
        detail: "offline",
        hint: "Start it with: .venv\\Scripts\\python -m uvicorn src.api.main:app --port 8000",
      });
    }

    try {
      const res = await fetch("/api/chat", { cache: "no-store" });
      const data = (await res.json()) as { configured?: boolean; model?: string };
      next.push({
        key: "chat",
        label: "Chat model key",
        state: data.configured ? "ok" : "warn",
        detail: data.configured ? (data.model ?? "configured") : "OPENAI_API_KEY not set",
        hint: "Add OPENAI_API_KEY to web/.env.local. Analyze mode and the presenter demo still work without it.",
      });
    } catch {
      next.push({ key: "chat", label: "Chat model key", state: "warn", detail: "could not check" });
    }

    try {
      const res = await fetch("/api/ml/index", { cache: "no-store" });
      const data = res.ok ? ((await res.json()) as { n_passages?: number }) : null;
      const n = data?.n_passages ?? 0;
      next.push({
        key: "index",
        label: "Document index",
        state: n > 0 ? "ok" : "warn",
        detail: n > 0 ? `${n} passages indexed` : "empty",
        hint: "Optional. Upload PDF, DOCX, or TXT files in Chat mode for document-grounded checks.",
      });
    } catch {
      next.push({ key: "index", label: "Document index", state: "warn", detail: "backend offline" });
    }

    setChecks(next);
    setBusy(false);
  }, [artifactsReady, modelsReady, resultsReady]);

  useEffect(() => {
    // deferred so the first setState happens outside the effect body
    queueMicrotask(() => void probe());
  }, [probe]);

  return (
    <section className="glass-panel rounded-2xl p-5 md:p-6 space-y-4" aria-label="Live demo readiness">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold">Live demo readiness</h2>
          <p className="text-[11px] text-muted-foreground">
            Preflight for the booth. Offline pages (/demo, /dashboard) never depend on these checks.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/analyze"
            className="hidden sm:inline-flex items-center gap-1.5 rounded-xl border border-border bg-secondary/60 px-3 py-2 text-xs font-semibold transition-colors hover:bg-secondary"
          >
            Open analyzer <ArrowRight className="h-3.5 w-3.5" aria-hidden />
          </Link>
          <button
            type="button"
            onClick={() => void probe()}
            disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-3 py-2 text-xs font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500 disabled:opacity-60"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} aria-hidden />
            Recheck
          </button>
        </div>
      </div>

      <ul className="grid gap-2 sm:grid-cols-2" aria-live="polite">
        {checks.map((check) => (
          <li key={check.key} className="surface-inset rounded-xl p-3.5">
            <div className="flex items-center gap-2">
              <StateIcon state={busy && checks.length === 0 ? "loading" : check.state} />
              <span className="text-xs font-semibold">{check.label}</span>
              <span className="ml-auto font-mono text-[10px] text-muted-foreground">{check.detail}</span>
            </div>
            {check.state !== "ok" && check.hint && (
              <p className="mt-1.5 text-[10px] leading-relaxed text-muted-foreground">{check.hint}</p>
            )}
          </li>
        ))}
        {checks.length === 0 && (
          <li className="surface-inset rounded-xl p-3.5 text-xs text-muted-foreground">Checking dependencies…</li>
        )}
      </ul>
    </section>
  );
}
