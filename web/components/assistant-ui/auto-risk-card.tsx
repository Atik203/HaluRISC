"use client";

import { useEffect, useRef, useState } from "react";
import { useAuiState } from "@assistant-ui/react";
import {
  AlertTriangle,
  ChevronDown,
  FileText,
  Loader2,
  MessagesSquare,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
} from "lucide-react";
import { buildAnalysisInput, CONTEXT_MAX, type Turn } from "@/lib/analysis-input";
import { useAutoAnalysis } from "@/components/assistant-ui/auto-analysis-context";

interface FeatureImpact {
  feature: string;
  value: number;
  impact: number;
}

interface AnalyzeResult {
  prediction: {
    calibrated_score: number;
    label: string;
    latency_ms?: number;
    model_version?: string;
    feature_version?: string;
    warning?: string;
    features?: Record<string, number>;
  };
  explanation?: { top_features?: FeatureImpact[]; base_value?: number } | null;
}

const LABEL_TEXT: Record<string, string> = {
  low_risk: "Low risk",
  medium_risk: "Medium risk",
  high_risk: "High risk",
};

function tone(label: string): { cls: string; Icon: typeof ShieldCheck } {
  if (label === "high_risk")
    return { cls: "border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400", Icon: ShieldAlert };
  if (label === "medium_risk")
    return { cls: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400", Icon: ShieldQuestion };
  return { cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400", Icon: ShieldCheck };
}

interface PartLike {
  type?: string;
  text?: string;
}

function extractText(message: { content?: readonly PartLike[] }): string {
  return (message.content ?? [])
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("\n")
    .trim();
}

function turnFromMessage(m: {
  id?: string;
  role?: string;
  content?: readonly PartLike[];
}): (Turn & { id: string }) | null {
  const text = extractText(m);
  if (!text || (m.role !== "user" && m.role !== "assistant") || !m.id) return null;
  return { role: m.role, text, id: m.id };
}

/**
 * B7.5 Tier 1 — automatic hallucination-risk card.
 * Rendered inside each assistant message; fires once the message completes:
 *   question = last user message, context = evidence panel or recent
 *   conversation, answer = this message. Toggle-off and skip rules respected.
 */
export function AutoRiskCard() {
  const { enabled, sessionContext } = useAutoAnalysis();
  const message = useAuiState((s) => s.message);
  const messages = useAuiState((s) => s.thread.messages);

  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [grounding, setGrounding] = useState<"evidence" | "conversation">("conversation");
  const analyzedRef = useRef<string | null>(null);

  const text = extractText(message);
  const isComplete = message?.status?.type === "complete";

  useEffect(() => {
    if (!enabled) return;
    if (message?.role !== "assistant") return;
    if (!isComplete) return;
    if (!text) return;

    const key = `${message.id}:${text.length}:${text.slice(0, 80)}`;
    if (analyzedRef.current === key) return;

    const turns = (messages ?? [])
      .map(turnFromMessage)
      .filter((t): t is Turn & { id: string } => t !== null);
    const answerIndex = turns.findIndex((t) => t.id === message.id);
    if (answerIndex < 0) return;
    const input = buildAnalysisInput(turns, answerIndex, sessionContext);
    if (!input) return;

    analyzedRef.current = key; // guard refires while the request is in flight
    let cancelled = false;

    (async () => {
      await Promise.resolve(); // defer setState out of the effect (no cascading renders)
      if (cancelled) return;
      setGrounding(input.grounding);
      setState("loading");
      try {
        const res = await fetch("/api/ml/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: input.question, context: input.context, answer: input.answer }),
        });
        if (!res.ok) {
          const detail = (await res.json().catch(() => null))?.detail;
          throw new Error(detail || `ML backend error (${res.status})`);
        }
        const data = (await res.json()) as AnalyzeResult;
        if (cancelled) return;
        setResult(data);
        setState("done");
      } catch {
        if (!cancelled) setState("error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [enabled, isComplete, message?.id, message?.role, text, messages, sessionContext]);

  if (!enabled) return null;
  if (message?.role !== "assistant" || !isComplete || !text) return null;

  if (state === "loading") {
    return (
      <div className="mt-3 flex items-center gap-3 rounded-xl border border-border/60 bg-secondary/30 px-4 py-3 text-xs text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin text-violet-500" aria-hidden />
        Checking hallucination risk…
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-[11px] text-amber-700 dark:text-amber-400" role="status">
        <AlertTriangle className="w-4 h-4" aria-hidden />
        Risk check unavailable — ML backend offline. /analyze and /demo still work.
      </div>
    );
  }

  if (state === "done" && result?.prediction) {
    const p = result.prediction;
    const score = Number(p.calibrated_score ?? 0);
    const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
    const { cls, Icon } = tone(p.label);
    const top = result.explanation?.top_features?.slice(0, 3) ?? [];
    const features = p.features ?? {};
    const groupCount = Object.keys(features).length;

    return (
      <section
        className="mt-3 rounded-xl border border-border/60 bg-secondary/20 p-4 space-y-3"
        aria-label={`Hallucination risk: ${LABEL_TEXT[p.label] ?? p.label}, ${pct} percent`}
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Icon className="w-6 h-6" aria-hidden />
            <div>
              <div className="text-sm font-bold">{LABEL_TEXT[p.label] ?? p.label}</div>
              <div className="text-[11px] font-mono text-muted-foreground">{pct}% calibrated risk probability</div>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 text-[10px] font-mono text-muted-foreground">
            {p.latency_ms != null && <span>{p.latency_ms.toFixed(0)} ms</span>}
            {p.model_version && <span>model {p.model_version}</span>}
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          {grounding === "evidence" ? (
            <>
              <FileText className="w-3.5 h-3.5" aria-hidden />
              Evidence: checked against your pasted context
            </>
          ) : (
            <>
              <MessagesSquare className="w-3.5 h-3.5" aria-hidden />
              Conversation only — no external grounding; scores reflect answer-vs-conversation consistency
            </>
          )}
        </div>

        {top.length > 0 && (
          <ul className="space-y-1 text-[11px]">
            {top.map((f) => (
              <li key={f.feature} className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground truncate">{f.feature}</span>
                <span className={`font-mono ${f.impact >= 0 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                  {f.impact >= 0 ? "+" : ""}
                  {f.impact.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        )}

        {p.warning && (
          <p className="text-[10px] text-amber-600/90 dark:text-amber-400/90 bg-amber-500/5 border border-amber-500/20 rounded-lg px-2.5 py-1.5">
            {p.warning}
          </p>
        )}

        {groupCount > 0 && (
          <details className="group">
            <summary className="flex items-center justify-between cursor-pointer text-[11px] font-semibold text-muted-foreground hover:text-foreground">
              <span>{groupCount} features · thresholds · provenance</span>
              <ChevronDown className="w-3.5 h-3.5 transition-transform group-open:rotate-180" aria-hidden />
            </summary>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-0.5 text-[10px]">
              {Object.entries(features).map(([name, value]) => (
                <div key={name} className="flex justify-between gap-3">
                  <span className="text-muted-foreground truncate">{name}</span>
                  <span className="font-mono">{Number(value).toFixed(4)}</span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground">
              SHAP values reflect the raw XGBoost model (feature attribution), not the calibrated score. Thresholds and
              provenance: see /analyze.
            </p>
          </details>
        )}
      </section>
    );
  }

  return null;
}
