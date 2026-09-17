"use client";

import {
  ActionBarPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { Bot, Check, Copy, Send, ShieldCheck, Sparkles, Square, User } from "lucide-react";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { AutoRiskCard } from "@/components/assistant-ui/auto-risk-card";

export interface Starter {
  label: string;
  title?: string;
  prompt: string;
}

function UserMessage() {
  return (
    <div className="flex items-end justify-end gap-2.5 animate-fade">
      <div className="max-w-2xl rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground shadow-sm">
        <MessagePrimitive.Content />
      </div>
      <div className="hidden sm:flex h-8 w-8 shrink-0 items-center justify-center rounded-xl surface-inset text-muted-foreground">
        <User className="h-4 w-4" aria-hidden />
      </div>
    </div>
  );
}

function MessageStatus() {
  const status = useAuiState((s) => s.message?.status?.type);
  if (status !== "running") return null;
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[10px] font-medium text-violet-600 dark:text-violet-300"
      role="status"
      aria-live="polite"
    >
      <span className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse-soft" aria-hidden />
      thinking
    </span>
  );
}

function AssistantMessage() {
  return (
    <div className="flex items-start gap-2.5 animate-fade">
      <div className="hidden sm:flex h-8 w-8 shrink-0 items-center justify-center rounded-xl surface-inset">
        <Bot className="h-4 w-4 text-violet-600 dark:text-purple-400" aria-hidden />
      </div>
      <div className="min-w-0 w-full max-w-3xl">
        <div className="overflow-hidden rounded-2xl rounded-tl-md border border-border/70 bg-card/70">
          <div className="flex items-center justify-between gap-3 border-b border-border/50 px-4 py-2">
            <span className="flex items-center gap-2 text-xs font-semibold">
              <ShieldCheck className="w-3.5 h-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
              HaluRISC
              <span className="text-[10px] font-normal text-muted-foreground">grounded risk analyst</span>
            </span>
            <span className="flex items-center gap-2">
              <MessageStatus />
              <ActionBarPrimitive.Copy
                title="Copy answer"
                aria-label="Copy answer"
                className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground data-[copied=true]:text-emerald-600 dark:data-[copied=true]:text-emerald-400"
              >
                <Copy className="h-3.5 w-3.5" aria-hidden />
              </ActionBarPrimitive.Copy>
            </span>
          </div>
          <div className="px-4 py-3.5">
            <MessagePrimitive.Content components={{ Text: MarkdownText }} />
          </div>
          <AutoRiskCard />
        </div>
      </div>
    </div>
  );
}

export function Thread({ starters = [] }: { starters?: Starter[] }) {
  const isRunning = useAuiState((s) => s.thread.isRunning);

  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col gap-3">
      <ThreadPrimitive.Viewport className="glass-panel flex-1 overflow-y-auto rounded-2xl p-4">
        <ThreadPrimitive.Empty>
          <div className="flex h-full flex-col items-center justify-center gap-7 py-6 text-center">
            <div className="space-y-2.5">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl surface-inset">
                <Sparkles className="h-5 w-5 text-violet-600 dark:text-purple-400 animate-pulse-soft" aria-hidden />
              </div>
              <h2 className="display-title text-xl">What should I check?</h2>
              <p className="mx-auto max-w-md text-sm text-muted-foreground leading-relaxed">
                Ask a question, paste an answer, and every response is scored against your evidence.
                Add documents or turn on web search for stronger grounding.
              </p>
            </div>
            <div className="grid w-full max-w-2xl gap-3 sm:grid-cols-2">
              {starters.map((s) => (
                <ThreadPrimitive.Suggestion
                  key={s.label}
                  prompt={s.prompt}
                  send
                  className="group h-full rounded-xl border border-border/70 bg-card/50 p-3.5 text-left transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:bg-secondary/40"
                >
                  <span className="block text-xs font-bold">{s.label}</span>
                  {s.title && (
                    <span className="mt-1 block text-[11px] leading-relaxed text-muted-foreground">{s.title}</span>
                  )}
                </ThreadPrimitive.Suggestion>
              ))}
            </div>
          </div>
        </ThreadPrimitive.Empty>
        <div className="space-y-4">
          <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
        </div>
      </ThreadPrimitive.Viewport>

      <ComposerPrimitive.Root className="glass-panel flex items-center gap-2 rounded-2xl p-2">
        <ComposerPrimitive.Input
          autoFocus
          placeholder="Paste a question, context, or answer to analyze…"
          className="flex-1 bg-transparent px-4 py-2 text-sm outline-none placeholder:text-muted-foreground/60"
        />
        {isRunning ? (
          <ComposerPrimitive.Cancel
            title="Stop generating"
            aria-label="Stop generating"
            className="p-2.5 rounded-xl border border-border bg-secondary text-foreground transition-colors hover:bg-secondary/70"
          >
            <Square className="h-4 w-4 fill-current" aria-hidden />
          </ComposerPrimitive.Cancel>
        ) : (
          <ComposerPrimitive.Send
            title="Send message"
            aria-label="Send message"
            className="rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 p-2.5 text-white transition-all hover:from-violet-500 hover:to-indigo-500 disabled:opacity-50"
          >
            <Send className="h-4 w-4" aria-hidden />
          </ComposerPrimitive.Send>
        )}
      </ComposerPrimitive.Root>

      <p className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground">
        <Check className="h-3 w-3" aria-hidden />
        Enter to send · Shift + Enter for a new line
      </p>
    </ThreadPrimitive.Root>
  );
}
