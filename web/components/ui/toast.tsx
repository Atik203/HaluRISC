"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Info } from "lucide-react";

type Tone = "success" | "info" | "error";

interface ToastItem {
  id: number;
  message: string;
  tone: Tone;
}

let items: ToastItem[] = [];
let listeners: Array<(next: ToastItem[]) => void> = [];
let nextId = 1;

function emit() {
  for (const listener of listeners) listener([...items]);
}

/** Fire-and-forget notification; usable from any client component. */
export function toast(message: string, tone: Tone = "success") {
  const item: ToastItem = { id: nextId++, message, tone };
  items = [...items, item];
  emit();
  window.setTimeout(() => {
    items = items.filter((entry) => entry.id !== item.id);
    emit();
  }, 3200);
}

const TONE_STYLES: Record<Tone, { cls: string; Icon: typeof Info }> = {
  success: { cls: "border-emerald-500/40 text-emerald-700 dark:text-emerald-300", Icon: CheckCircle2 },
  info: { cls: "border-border text-foreground", Icon: Info },
  error: { cls: "border-rose-500/40 text-rose-700 dark:text-rose-300", Icon: AlertTriangle },
};

export function Toaster() {
  const [list, setList] = useState<ToastItem[]>([]);

  useEffect(() => {
    const listener = (next: ToastItem[]) => setList(next);
    listeners.push(listener);
    listener(items);
    return () => {
      listeners = listeners.filter((entry) => entry !== listener);
    };
  }, []);

  if (list.length === 0) return null;

  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-[120] flex w-full max-w-xs flex-col gap-2 print:hidden"
      role="status"
      aria-live="polite"
    >
      {list.map((item) => {
        const { cls, Icon } = TONE_STYLES[item.tone];
        return (
          <div
            key={item.id}
            className={`glass-panel pointer-events-auto flex items-start gap-2 rounded-xl border px-3.5 py-2.5 text-xs font-medium shadow-lg animate-rise ${cls}`}
          >
            <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="leading-relaxed">{item.message}</span>
          </div>
        );
      })}
    </div>
  );
}
