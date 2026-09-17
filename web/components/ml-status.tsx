"use client";

import { useEffect, useState } from "react";

type Status = "checking" | "connected" | "degraded" | "offline";

/**
 * Real backend health indicator: polls the FastAPI /health endpoint
 * (via the Next.js /api/ml rewrite). Replaces the old static "Connected" dot.
 */
export function MlStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [detail, setDetail] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const check = async () => {
      try {
        const res = await fetch("/api/ml/health", { cache: "no-store" });
        if (!res.ok) throw new Error(`http ${res.status}`);
        const data = (await res.json()) as {
          status?: string;
          model?: string;
          artifacts_loaded?: boolean;
          feature_models_ready?: boolean;
        };
        if (cancelled) return;
        if (data.status === "ok" && data.artifacts_loaded) {
          setStatus("connected");
          setDetail(data.model ?? "");
        } else {
          setStatus("degraded");
          setDetail(data.status ?? "");
        }
      } catch {
        if (!cancelled) {
          setStatus("offline");
          setDetail("");
        }
      }
    };

    check();
    timer = setTimeout(function tick() {
      check();
      timer = setTimeout(tick, 30_000);
    }, 30_000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const config = {
    checking: { color: "bg-amber-400", label: "Checking backend…" },
    connected: { color: "bg-emerald-500", label: "ML backend connected" },
    degraded: { color: "bg-amber-500", label: "ML backend degraded" },
    offline: { color: "bg-rose-500", label: "ML backend offline — use /analyze or /demo" },
  }[status];

  return (
    <span className="inline-flex items-center gap-2" role="status" aria-live="polite">
      <span className={`w-2.5 h-2.5 rounded-full ${config.color} ${status === "checking" ? "animate-pulse" : ""}`} aria-hidden />
      <span className="text-xs font-mono text-muted-foreground">{config.label}</span>
      {detail && <span className="text-[10px] font-mono text-muted-foreground/70 hidden md:inline">{detail}</span>}
    </span>
  );
}
