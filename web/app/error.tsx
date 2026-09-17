"use client";

import Link from "next/link";
import { AlertTriangle, RotateCcw, Home } from "lucide-react";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="glass-panel rounded-2xl p-8 max-w-lg w-full text-center space-y-4 animate-rise">
        <div className="w-12 h-12 mx-auto rounded-2xl surface-inset flex items-center justify-center">
          <AlertTriangle className="w-6 h-6 text-amber-600 dark:text-amber-400" aria-hidden />
        </div>
        <h1 className="display-title text-2xl">Something went wrong</h1>
        <p className="text-sm text-muted-foreground">
          This view could not be rendered. The ML backend and artifacts are unaffected.
        </p>
        {error.digest && (
          <p className="text-[11px] font-mono text-muted-foreground/80">digest {error.digest}</p>
        )}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500"
          >
            <RotateCcw className="w-4 h-4" aria-hidden /> Try again
          </button>
          <Link
            href="/"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary"
          >
            <Home className="w-4 h-4" aria-hidden /> Back to start
          </Link>
        </div>
        <details className="text-left pt-2">
          <summary className="cursor-pointer text-xs font-semibold text-muted-foreground hover:text-foreground">
            Technical details
          </summary>
          <pre className="mt-2 max-h-40 overflow-auto rounded-xl surface-inset p-3 text-[11px] font-mono whitespace-pre-wrap">
            {error.message}
          </pre>
        </details>
      </div>
    </div>
  );
}
