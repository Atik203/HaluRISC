"use client";

import { useEffect, useRef, useState } from "react";
import { useAuiState } from "@assistant-ui/react";
import {
  AlertTriangle,
  ChevronDown,
  FileText,
  Globe,
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

interface ClaimVerdict {
  id: number;
  text: string;
  verdict: "supported" | "contradicted" | "unsupported";
  confidence: number;
  evidence_sentence: string;
  evidence_quote?: string;
  evidence_source?: string;
  evidence_url?: string;
  abstained?: boolean;
  judged_by?: string;
  judge_reasoning?: string;
}

interface VerifyResult extends AnalyzeResult {
  claims?: ClaimVerdict[];
  aggregate?: {
    n: number;
    supported: number;
    contradicted: number;
    unsupported: number;
    overall: string;
    abstained?: number;
    evidence_mode?: string;
    llm_judged?: number;
  };
}

/** T4: 👍/👎 feedback on a verdict; posts to /api/ml/feedback. */
function FeedbackButtons({
  payload,
}: {
  payload: () => Record<string, string> | null;
}) {
  const [sent, setSent] = useState<"agree" | "disagree" | null>(null);
  const send = async (feedback: "agree" | "disagree") => {
    if (sent) return;
    setSent(feedback);
    const body = payload();
    if (!body) return;
    try {
      await fetch("/api/ml/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, feedback }),
      });
    } catch {
      /* feedback is best-effort */
    }
  };
  return (
    <span className="inline-flex items-center gap-1 ml-2 align-middle">
      <button
        type="button"
        onClick={() => send("agree")}
        aria-label="Agree with this verdict"
        className={`text-xs rounded-md border px-1.5 py-0.5 transition-colors ${
          sent === "agree"
            ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-500"
            : "border-border/60 text-muted-foreground hover:text-emerald-500 hover:border-emerald-500/40"
        }`}
      >
        👍
      </button>
      <button
        type="button"
        onClick={() => send("disagree")}
        aria-label="Disagree with this verdict"
        className={`text-xs rounded-md border px-1.5 py-0.5 transition-colors ${
          sent === "disagree"
            ? "border-rose-500/50 bg-rose-500/10 text-rose-500"
            : "border-border/60 text-muted-foreground hover:text-rose-500 hover:border-rose-500/40"
        }`}
      >
        👎
      </button>
    </span>
  );
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

const CLAIM_TONE: Record<string, string> = {
  supported: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  contradicted: "border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400",
  unsupported: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
};

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
  const { enabled, sessionContext, webEnabled, hasDocuments } = useAutoAnalysis();
  const message = useAuiState((s) => s.message);
  const messages = useAuiState((s) => s.thread.messages);

  const [state, setState] = useState<"idle" | "loading" | "done" | "error" | "timeout">("idle");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [grounding, setGrounding] = useState<"evidence" | "conversation">("conversation");
  const analyzedRef = useRef<string | null>(null);
  const inputRef = useRef<{ question: string; context: string; answer: string } | null>(null);
  const unmountedRef = useRef(false);

  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true; // only unmount cancels; store updates must not
    };
  }, []);

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
    inputRef.current = { question: input.question, context: input.context, answer: input.answer };

    (async () => {
      await Promise.resolve(); // defer setState out of the effect (no cascading renders)
      if (unmountedRef.current) return;
      setGrounding(input.grounding);
      setState("loading");
      const payload = JSON.stringify({
        question: input.question,
        context: input.context,
        answer: input.answer,
        evidence_mode: hasDocuments || webEnabled ? "auto" : "context",
      });
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 120_000); // hang -> error card
      try {
        // Tier 2 first: claim-level verification (includes the prediction).
        // Falls back to the Tier-1 combined analyze when unavailable.
        let res = await fetch("/api/ml/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          signal: controller.signal,
        });
        if (!res.ok) {
          res = await fetch("/api/ml/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: payload,
            signal: controller.signal,
          });
        }
        if (!res.ok) {
          const detail = (await res.json().catch(() => null))?.detail;
          throw new Error(detail || `ML backend error (${res.status})`);
        }
        const data = (await res.json()) as VerifyResult;
        clearTimeout(timeout);
        if (unmountedRef.current) return;
        setResult(data);
        setState("done");
      } catch (err) {
        clearTimeout(timeout);
        if (unmountedRef.current) return;
        setState(err instanceof DOMException && err.name === "AbortError" ? "timeout" : "error");
      }
    })();
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

  if (state === "timeout") {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-[11px] text-amber-700 dark:text-amber-400" role="status">
        <AlertTriangle className="w-4 h-4" aria-hidden />
        Risk check timed out — is the ML backend running (uvicorn on port 8000)? /analyze and /demo still work.
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
    const claims = result.claims ?? [];
    const agg = result.aggregate;

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
          {(() => {
            const mode = agg?.evidence_mode ?? (grounding === "evidence" ? "context" : "conversation");
            if (mode === "index") {
              return (
                <>
                  <FileText className="w-3.5 h-3.5" aria-hidden />
                  Evidence: your uploaded documents (cited per claim)
                </>
              );
            }
            if (mode === "web") {
              return (
                <>
                  <Globe className="w-3.5 h-3.5" aria-hidden />
                  Evidence: live web search (cited per claim)
                </>
              );
            }
            if (mode === "context" || grounding === "evidence") {
              return (
                <>
                  <FileText className="w-3.5 h-3.5" aria-hidden />
                  Evidence: checked against your pasted context
                </>
              );
            }
            return (
              <>
                <MessagesSquare className="w-3.5 h-3.5" aria-hidden />
                Conversation only — no external grounding; scores reflect answer-vs-conversation consistency
              </>
            );
          })()}
        </div>

        {claims.length > 0 && agg && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="font-semibold text-muted-foreground">Per-claim verdicts (NLI):</span>
              <span className={`px-2 py-0.5 rounded-full border font-mono ${CLAIM_TONE.supported}`}>{agg.supported} supported</span>
              <span className={`px-2 py-0.5 rounded-full border font-mono ${CLAIM_TONE.contradicted}`}>{agg.contradicted} contradicted</span>
              <span className={`px-2 py-0.5 rounded-full border font-mono ${CLAIM_TONE.unsupported}`}>{agg.unsupported} unsupported</span>
              {(agg.abstained ?? 0) > 0 && (
                <span className="px-2 py-0.5 rounded-full border border-border bg-secondary/40 font-mono text-muted-foreground">
                  {agg.abstained} abstained (no evidence)
                </span>
              )}
              {(agg.llm_judged ?? 0) > 0 && (
                <span className="px-2 py-0.5 rounded-full border border-indigo-500/40 bg-indigo-500/10 font-mono text-indigo-600 dark:text-indigo-400">
                  {agg.llm_judged} LLM-judged
                </span>
              )}
              <FeedbackButtons
                payload={() => {
                  const input = inputRef.current;
                  if (!input) return null;
                  return {
                    question: input.question, context: input.context, answer: input.answer,
                    claim_text: "", verdict: agg.overall,
                    evidence_sentence: "", inputs_hash: "",
                  };
                }}
              />
            </div>
            <ul className="space-y-1.5">
              {claims.map((c) => (
                <li key={c.id} className="text-[11px]">
                  <span className={`inline-block px-2 py-0.5 rounded-full border font-mono mr-2 ${CLAIM_TONE[c.verdict]}`}>
                    {c.verdict}
                  </span>
                  <span className="text-foreground/90">{c.text}</span>
                  {c.judged_by === "llm" && c.judge_reasoning && (
                    <span className="ml-1.5 text-[10px] text-indigo-600 dark:text-indigo-400 not-italic">
                      LLM: {c.judge_reasoning.length > 90 ? `${c.judge_reasoning.slice(0, 90)}…` : c.judge_reasoning}
                    </span>
                  )}
                  <FeedbackButtons
                    payload={() => {
                      const input = inputRef.current;
                      if (!input) return null;
                      return {
                        question: input.question, context: input.context, answer: input.answer,
                        claim_text: c.text, verdict: c.verdict,
                        evidence_sentence: c.evidence_sentence, inputs_hash: "",
                      };
                    }}
                  />
                  {c.abstained ? (
                    <p className="mt-0.5 pl-1 text-[10px] text-amber-600/90 dark:text-amber-400/90">
                      no evidence retrieved — abstained
                    </p>
                  ) : (
                    <>
                      {c.verdict === "contradicted" && c.evidence_quote && (
                        <p className="mt-0.5 pl-1 text-[10px] text-rose-700/90 dark:text-rose-400/90">
                          <span className="font-semibold not-italic">Evidence says:</span>{" "}
                          {c.evidence_quote.length > 200 ? `${c.evidence_quote.slice(0, 200)}…` : c.evidence_quote}
                        </p>
                      )}
                      <p className="mt-0.5 pl-1 text-[10px] text-muted-foreground italic">
                        evidence: {c.evidence_sentence.length > 140 ? `${c.evidence_sentence.slice(0, 140)}…` : c.evidence_sentence}
                        {c.evidence_source?.startsWith("web:") && c.evidence_url ? (
                          <a
                            href={c.evidence_url}
                            target="_blank"
                            rel="noreferrer"
                            className="ml-1.5 inline-flex items-center gap-0.5 text-violet-600 dark:text-purple-400 not-italic underline"
                          >
                            <Globe className="w-3 h-3" aria-hidden /> source
                          </a>
                        ) : c.evidence_source?.startsWith("doc:") ? (
                          <span className="ml-1.5 not-italic text-muted-foreground/80">· {c.evidence_source.replace("doc:", "")}</span>
                        ) : null}
                      </p>
                    </>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

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
