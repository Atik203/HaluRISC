export default function Loading() {
  return (
    <div className="min-h-[50vh] flex flex-col items-center justify-center gap-4" role="status" aria-live="polite">
      <div className="flex items-center gap-1.5" aria-hidden>
        <span className="w-2 h-2 rounded-full bg-violet-500 animate-pulse-soft" />
        <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse-soft [animation-delay:160ms]" />
        <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse-soft [animation-delay:320ms]" />
      </div>
      <p className="text-xs text-muted-foreground font-mono">Loading view…</p>
    </div>
  );
}
