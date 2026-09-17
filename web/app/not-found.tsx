import Link from "next/link";
import { Compass, MessageSquare, BarChart2 } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="glass-panel rounded-2xl p-8 max-w-lg w-full text-center space-y-4 animate-rise">
        <p className="display-title text-6xl gradient-text tnum">404</p>
        <h1 className="text-lg font-semibold">This page does not exist</h1>
        <p className="text-sm text-muted-foreground">
          The address may be stale, or the view was renamed. Pick a destination below.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500"
          >
            <MessageSquare className="w-4 h-4" aria-hidden /> Chat mode
          </Link>
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary"
          >
            <BarChart2 className="w-4 h-4" aria-hidden /> Analyze mode
          </Link>
          <Link
            href="/demo"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary"
          >
            <Compass className="w-4 h-4" aria-hidden /> Presenter demo
          </Link>
        </div>
      </div>
    </div>
  );
}
