"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Maximize2, X } from "lucide-react";

/** Figure with a click-to-zoom overlay, useful when presenting charts on a booth screen. */
export function FigureLightbox({
  src,
  alt,
  className = "",
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close]);

  return (
    <>
      <div className={`group relative ${className}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          loading="lazy"
          className="w-full rounded-xl border border-border/60 bg-secondary/30"
        />
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label={`Zoom: ${alt}`}
          className="absolute right-2 top-2 inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-background/85 px-2 py-1 text-[10px] font-semibold text-muted-foreground opacity-0 backdrop-blur transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
        >
          <Maximize2 className="h-3 w-3" aria-hidden /> Zoom
        </button>
      </div>

      {open && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-10 print:hidden"
          role="dialog"
          aria-modal="true"
          aria-label={alt}
        >
          <button
            type="button"
            aria-label="Close zoomed figure"
            onClick={close}
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
          />
          <div className="relative flex max-h-full w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
            <div className="flex items-center justify-between gap-3 border-b border-border/50 px-4 py-2.5">
              <p className="text-xs text-muted-foreground">{alt}</p>
              <button
                ref={closeRef}
                type="button"
                onClick={close}
                aria-label="Close"
                className="rounded-lg border border-border bg-secondary/60 p-1.5 transition-colors hover:bg-secondary"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-auto bg-background p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src} alt={alt} className="w-full rounded-lg" />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
