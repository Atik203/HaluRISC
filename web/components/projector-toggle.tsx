"use client";

import { useSyncExternalStore } from "react";
import { Projector } from "lucide-react";

const KEY = "halurisc-projector";

function subscribe(onChange: () => void) {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

function getSnapshot(): string {
  try {
    return localStorage.getItem(KEY) ?? "off";
  } catch {
    return "off";
  }
}

/** Scales the whole interface up for projector / booth screens. */
export function ProjectorToggle() {
  const mode = useSyncExternalStore(subscribe, getSnapshot, () => "off");
  const on = mode === "on";

  const toggle = () => {
    const next = on ? "off" : "on";
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* ignore */
    }
    document.documentElement.classList.toggle("projector", next === "on");
    window.dispatchEvent(new Event("storage"));
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={on}
      aria-label={on ? "Disable projector view" : "Enable projector view"}
      title={on ? "Projector view on: larger text" : "Projector view: larger text for big screens"}
      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border transition-all ${
        on
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border bg-secondary/50 text-muted-foreground hover:bg-secondary hover:text-foreground"
      }`}
    >
      <Projector className="h-4 w-4" aria-hidden />
    </button>
  );
}
