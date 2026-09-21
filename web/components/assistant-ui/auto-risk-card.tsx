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
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { buildAnalysisInput, type Turn } from "@/lib/analysis-input";
import { useAutoAnalysis } from "@/components/assistant-ui/auto-analysis-context";
import { Term } from "@/components/ui/term";
import { toast } from "@/components/ui/toast";

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
    thresholds?: { low?: number; medium?: number };
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

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/** T4: thumbs feedback on a verdict; posts to /api/ml/feedback. */
function FeedbackButtons({ payload }: { payload: () => Record<string, string> | null }) {
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
      toast("Feedback saved locally");
    } catch {
      /* feedback is best-effort */
    }
  };
  const base =
    "inline-flex h-6 w-6 items-center justify-center rounded-md border transition-colors";
  return (
    <span className="inline-flex shrink-0 items-center gap-1" aria-label="Verdict feedback">
      <button
        type="button"
        onClick={() => send("agree")}
        aria-label="Agree with this verdict"
        aria-pressed={sent === "agree"}
        className={`${base} ${
          sent === "agree"
            ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
            : "border-border/60 text-muted-foreground hover:border-emerald-500/40 hover:text-emerald-600 dark:hover:text-emerald-400"
        }`}
      >
        <ThumbsUp className="h-3 w-3" aria-hidden />
      </button>
      <button
        type="button"
        onClick={() => send("disagree")}
        aria-label="Disagree with this verdict"
        aria-pressed={sent === "disagree"}
        className={`${base} ${
          sent === "disagree"
            ? "border-rose-500/50 bg-rose-500/10 text-rose-600 dark:text-rose-400"
            : "border-border/60 text-muted-foreground hover:border-rose-500/40 hover:text-rose-600 dark:hover:text-rose-400"
        }`}
      >
        <ThumbsDown className="h-3 w-3" aria-hidden />
      </button>
    </span>
  );
}

function tone(label: string): {
  cls: string;
  bar: string;
  headline: string;
  Icon: typeof ShieldCheck;
} {
  if (label === "high_risk")
    return {
      cls: "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300",
      bar: "var(--risk-high)",
      headline: "Likely hallucinated",
      Icon: ShieldAlert,
    };
  if (label === "medium_risk")
    return {
      cls: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
      bar: "var(--risk-medium)",
      headline: "Needs a closer look",
      Icon: ShieldQuestion,
    };
  return {
    cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
      bar: "var(--risk-low)",
    headline: "Grounded in the evidence",
    Icon: ShieldCheck,
  };
}

interface PartLike {
  type?: string;
  text?: string;
}

const CLAIM_TONE: Record<string, string> = {
  supported: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  contradicted: "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300",
  unsupported: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
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
      <div className="border-t border-border/50 px-4 py-3" role="status" aria-live="polite">
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-violet-500" aria-hidden />
          {grounding === "evidence"
            ? "Checking each claim against your evidence…"
            : "Checking the answer against the conversation…"}
        </div>
        <div className="mt-2.5 space-y-1.5" aria-hidden>
          <div className="h-2 w-3/4 rounded-full bg-secondary/70 animate-pulse-soft" />
          <div className="h-2 w-1/2 rounded-full bg-secondary/70 animate-pulse-soft [animation-delay:150ms]" />
        </div>
      </div>
    );
  }

  if (state === "timeout" || state === "error") {
    return (
      <div className="border-t border-border/50 px-4 py-3">
        <div
          className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 px-3 py-2.5 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300"
          role="status"
        >
          <AlertTriangle className="mt-0.5 w-3.5 h-3.5 shrink-0" aria-hidden />
          <span>
            {state === "timeout"
              ? "The risk check timed out. The local ML service may still be starting."
              : "The risk check could not reach the local ML service."}{" "}
            Analyze mode and the offline presenter demo still work.
          </span>
        </div>
      </div>
    );
  }

  if (state === "done" && result?.prediction) {
    const p = result.prediction;
    const score = Number(p.calibrated_score ?? 0);
    const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
    const top = result.explanation?.top_features?.slice(0, 3) ?? [];
    const features = p.features ?? {};
    const groupCount = Object.keys(features).length;
    const claims = result.claims ?? [];
    const agg = result.aggregate;
    // Primary signal: per-claim evidence verdicts. The calibrated XGBoost
    // score is style-sensitive (HaluEval favors terse answers) and only leads
    // when no claim verdicts exist.
    const hasClaims = claims.length > 0;
    const claimRisk = hasClaims && agg
      ? (agg.contradicted ?? 0) > 0
        ? "high_risk"
        : (agg.unsupported ?? 0) > 0
          ? "medium_risk"
          : "low_risk"
      : null;
    const headlineLabel = claimRisk ?? p.label;
    const { cls, bar, headline, Icon } = tone(headlineLabel);
    const legacyConflict = claimRisk != null && claimRisk !== "high_risk" && score >= 0.6;
    const maxAbs = top.length > 0 ? Math.max(...top.map((f) => Math.abs(f.impact)), 1e-6) : 1;
    const thresholds = p.thresholds;

    return (
      <section
        className="border-t border-border/50 px-4 py-4 space-y-3.5"
        aria-label={`Hallucination risk: ${headline}, ${pct} percent`}
      >
        {/* Verdict banner */}
        <div className={`rounded-xl border p-3.5 ${cls}`}>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-background/40">
              <Icon className="h-5 w-5" aria-hidden />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-bold leading-tight">{headline}</p>
              <p className="mt-0.5 text-[11px] leading-snug opacity-80">
                {claimRisk
                  ? `${claims.length} claim${claims.length > 1 ? "s" : ""} checked · headline follows the claim evidence`
                  : "Evidence-calibrated risk probability"}
              </p>
            </div>
            <div className="text-right">
              <div className="font-mono text-2xl font-extrabold tnum leading-none">{pct}%</div>
              <div className="mt-0.5 font-mono text-[10px] opacity-70">{claimRisk ? "evidence score" : "risk"}</div>
            </div>
          </div>

          {/* Score meter with decision thresholds */}
          <div className="relative mt-3 h-1.5 overflow-hidden rounded-full bg-background/50">
            <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${pct}%`, backgroundColor: bar }} />
            {thresholds?.low != null && (
              <span
                className="absolute inset-y-0 w-px bg-background/80"
                style={{ left: `${thresholds.low * 100}%` }}
                aria-hidden
              />
            )}
            {thresholds?.medium != null && (
              <span
                className="absolute inset-y-0 w-px bg-background/80"
                style={{ left: `${thresholds.medium * 100}%` }}
                aria-hidden
              />
            )}
          </div>
          {thresholds?.low != null && thresholds.medium != null && (
            <p className="mt-1.5 font-mono text-[10px] opacity-70">
              low &lt; {thresholds.low.toFixed(2)} · medium {thresholds.low.toFixed(2)}–{thresholds.medium.toFixed(2)} · high ≥{" "}
              {thresholds.medium.toFixed(2)}
            </p>
          )}
        </div>

        {/* Grounding mode + provenance */}
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            {(() => {
              const mode = agg?.evidence_mode ?? (grounding === "evidence" ? "context" : "conversation");
              if (mode === "index") {
                return (
                  <>
                    <FileText className="w-3.5 h-3.5" aria-hidden />
                    Evidence: your uploaded documents, cited per claim
                  </>
                );
              }
              if (mode === "web") {
                return (
                  <>
                    <Globe className="w-3.5 h-3.5" aria-hidden />
                    Evidence: live web search, cited per claim
                  </>
                );
              }
              if (mode === "context" || grounding === "evidence") {
                return (
                  <>
                    <FileText className="w-3.5 h-3.5" aria-hidden />
                    Evidence: your pasted context
                  </>
                );
              }
              return (
                <>
                  <MessagesSquare className="w-3.5 h-3.5" aria-hidden />
                  Conversation only, no external grounding
                </>
              );
            })()}
          </span>
          <span className="font-mono">
            {p.latency_ms != null && <>{p.latency_ms.toFixed(0)} ms</>}
            {p.model_version && <> · model {p.model_version}</>}
          </span>
        </div>

        {legacyConflict && (
          <p className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-2.5 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
            The model score is style-sensitive because it was trained on synthetic HaluEval data. The headline here follows
            the claim evidence, so trust the verdict and the citations first.
          </p>
        )}

        {/* Claim verdicts */}
        {hasClaims && agg && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="eyebrow">Claim verdicts</span>
              <span className={`rounded-md border px-1.5 py-0.5 font-mono text-[10px] ${CLAIM_TONE.supported}`}>
                {agg.supported} supported
              </span>
              <span className={`rounded-md border px-1.5 py-0.5 font-mono text-[10px] ${CLAIM_TONE.contradicted}`}>
                {agg.contradicted} contradicted
              </span>
              <span className={`rounded-md border px-1.5 py-0.5 font-mono text-[10px] ${CLAIM_TONE.unsupported}`}>
                {agg.unsupported} unsupported
              </span>
              {(agg.abstained ?? 0) > 0 && (
                <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {agg.abstained} not judged
                </span>
              )}
              {(agg.llm_judged ?? 0) > 0 && (
                <span className="rounded-md border border-indigo-500/40 bg-indigo-500/10 px-1.5 py-0.5 font-mono text-[10px] text-indigo-600 dark:text-indigo-300">
                  {agg.llm_judged} LLM-reviewed
                </span>
              )}
              <span className="ml-auto">
                <FeedbackButtons
                  payload={() => {
                    const input = inputRef.current;
                    if (!input) return null;
                    return {
                      question: input.question,
                      context: input.context,
                      answer: input.answer,
                      claim_text: "",
                      verdict: agg.overall,
                      evidence_sentence: "",
                      inputs_hash: "",
                    };
                  }}
                />
              </span>
            </div>

            <ul className="space-y-2">
              {claims.map((c) => (
                <li key={c.id} className="rounded-xl border border-border/60 bg-card/40 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <span
                        className={`inline-block rounded-md border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${CLAIM_TONE[c.verdict]}`}
                      >
                        {c.verdict}
                      </span>
                      <p className="mt-1.5 text-xs leading-relaxed text-foreground/90">{c.text}</p>
                    </div>
                    <FeedbackButtons
                      payload={() => {
                        const input = inputRef.current;
                        if (!input) return null;
                        return {
                          question: input.question,
                          context: input.context,
                          answer: input.answer,
                          claim_text: c.text,
                          verdict: c.verdict,
                          evidence_sentence: c.evidence_sentence,
                          inputs_hash: "",
                        };
                      }}
                    />
                  </div>

                  {c.abstained ? (
                    <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">
                      No evidence was retrieved, so this claim was not judged.
                    </p>
                  ) : (
                    <>
                      {c.verdict === "contradicted" && c.evidence_quote && (
                        <blockquote className="mt-2 rounded-lg border-l-2 border-rose-500/60 bg-rose-500/5 px-2.5 py-1.5 text-[11px] leading-relaxed text-rose-700 dark:text-rose-300">
                          <span className="font-semibold">Evidence says: </span>
                          {truncate(c.evidence_quote, 220)}
                        </blockquote>
                      )}
                      {c.evidence_sentence && (
                        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
                          {truncate(c.evidence_sentence, 150)}
                          {c.evidence_source?.startsWith("web:") && c.evidence_url ? (
                            <a
                              href={c.evidence_url}
                              target="_blank"
                              rel="noreferrer"
                              className="ml-1.5 inline-flex items-center gap-0.5 text-violet-600 underline dark:text-purple-400"
                            >
                              <Globe className="h-3 w-3" aria-hidden /> source
                            </a>
                          ) : c.evidence_source?.startsWith("doc:") ? (
                            <span className="ml-1.5 text-muted-foreground/80">· {c.evidence_source.replace("doc:", "")}</span>
                          ) : null}
                        </p>
                      )}
                    </>
                  )}

                  {c.judged_by === "llm" && c.judge_reasoning && (
                    <p className="mt-1.5 text-[11px] leading-relaxed text-indigo-600 dark:text-indigo-300">
                      LLM review: {truncate(c.judge_reasoning, 140)}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Feature attribution */}
        {top.length > 0 && (
          <div className="space-y-2">
            <span className="eyebrow">Why this score</span>
            <ul className="space-y-2">
              {top.map((f) => (
                <li key={f.feature} className="space-y-1">
                  <div className="flex items-center justify-between gap-3 text-[11px]">
                    <span className="truncate text-muted-foreground">{f.feature}</span>
                    <span
                      className={`font-mono tnum ${
                        f.impact >= 0 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"
                      }`}
                    >
                      {f.impact >= 0 ? "+" : ""}
                      {f.impact.toFixed(3)}
                    </span>
                  </div>
                  <div className="relative h-1.5 rounded-full bg-secondary/70">
                    <span className="absolute inset-y-0 left-1/2 w-px bg-border" aria-hidden />
                    <span
                      className={`absolute inset-y-0 rounded-full ${f.impact >= 0 ? "left-1/2 bg-rose-500/80" : "right-1/2 bg-emerald-500/80"}`}
                      style={{ width: `${(Math.abs(f.impact) / maxAbs) * 50}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              <Term def="SHAP assigns each feature a contribution to the model output.">SHAP</Term> values explain the raw
              XGBoost model, not the calibrated score. Red raises risk, green lowers it.
            </p>
          </div>
        )}

        {p.warning && (
          <p className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-2.5 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
            {p.warning}
          </p>
        )}

        {groupCount > 0 && (
          <details className="group">
            <summary className="flex cursor-pointer items-center justify-between text-[11px] font-semibold text-muted-foreground hover:text-foreground">
              <span>{groupCount} features used for this score</span>
              <ChevronDown className="w-3.5 h-3.5 transition-transform group-open:rotate-180" aria-hidden />
            </summary>
            <div className="mt-2 grid grid-cols-1 gap-x-6 gap-y-0.5 text-[10px] sm:grid-cols-2">
              {Object.entries(features).map(([name, value]) => (
                <div key={name} className="flex justify-between gap-3">
                  <span className="truncate text-muted-foreground">{name}</span>
                  <span className="font-mono tnum">{Number(value).toFixed(4)}</span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">
              Full thresholds and provenance live in Analyze mode.
            </p>
          </details>
        )}

        <details className="group">
          <summary className="flex cursor-pointer items-center justify-between text-[11px] font-semibold text-muted-foreground hover:text-foreground">
            <span>How to read this card</span>
            <ChevronDown className="w-3.5 h-3.5 transition-transform group-open:rotate-180" aria-hidden />
          </summary>
          <ul className="mt-2 space-y-1.5 text-[10px] leading-relaxed text-muted-foreground">
            <li>
              <span className="font-semibold text-foreground">Headline.</span> When claims were checked, the claim
              evidence decides the verdict. The percentage is the calibrated model score shown for reference, and the
              two can disagree.
            </li>
            <li>
              <span className="font-semibold text-foreground">Meter.</span> The small ticks mark the low, medium, and
              high cutoffs that the API uses to pick the label.
            </li>
            <li>
              <span className="font-semibold text-foreground">Claim verdicts.</span> Supported means the evidence backs
              the claim. Contradicted means the evidence says the opposite. Unsupported means the evidence does not
              speak to it. Not judged means no evidence was retrieved.
            </li>
            <li>
              <span className="font-semibold text-foreground">Why this score.</span> The bars are SHAP contributions
              from the raw model, so they show what moved the score, not whether the answer is true.
            </li>
            <li>
              <span className="font-semibold text-foreground">Thumbs.</span> Feedback is appended to a local file on
              this machine only. Nothing is sent to an external service.
            </li>
          </ul>
        </details>
      </section>
    );
  }

  return null;
}
