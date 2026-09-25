"use client";

import { useState } from "react";
import {
  ActionBarPrimitive,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { Bot, Check, Copy, Send, Sparkles, Square, User } from "lucide-react";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { AutoRiskCard } from "@/components/assistant-ui/auto-risk-card";
import { ComposerChips, ComposerControls } from "@/components/chat/composer-toolbar";
import { useChatControls } from "@/components/chat/chat-controls-context";

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
      <div className="hidden sm:flex h-7 w-7 shrink-0 items-center justify-center rounded-lg surface-inset text-muted-foreground">
        <User className="h-3.5 w-3.5" aria-hidden />
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
    <div className="flex items-start gap-3 animate-fade">
      <div className="hidden sm:flex h-7 w-7 shrink-0 items-center justify-center rounded-lg surface-inset">
        <Bot className="h-3.5 w-3.5 text-violet-600 dark:text-purple-400" aria-hidden />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-xs font-semibold">
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
        <div className="mt-1.5 text-sm leading-relaxed">
          <MessagePrimitive.Content components={{ Text: MarkdownText }} />
        </div>
        <AutoRiskCard />
      </div>
    </div>
  );
}

export function Thread({ starters = [] }: { starters?: Starter[] }) {
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const { onUpload } = useChatControls();
  const [dragging, setDragging] = useState(false);

  return (
    <ThreadPrimitive.Root className="relative flex min-h-0 flex-1 flex-col">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto scroll-smooth">
        <div className="mx-auto flex w-full max-w-3xl flex-col px-4 pb-10 pt-6 md:px-6">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <div className="flex flex-col items-center gap-7 py-10 text-center">
              <div className="space-y-2.5">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl surface-inset">
                  <Sparkles className="h-5 w-5 text-violet-600 dark:text-purple-400 animate-pulse-soft" aria-hidden />
                </div>
                <h2 className="display-title text-xl">What should I check?</h2>
                <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
                  Ask a question, paste an answer, and every response is scored against your evidence. Add files or
                  reference text from the composer, or turn on web search.
                </p>
              </div>
              <div className="grid w-full gap-3 sm:grid-cols-2">
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
          </AuiIf>

          <div className="space-y-6">
            <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
          </div>
        </div>
      </ThreadPrimitive.Viewport>

      <div className="relative shrink-0 px-4 pb-4 md:px-6">
        <div
          className="pointer-events-none absolute inset-x-0 -top-12 h-12 bg-gradient-to-t from-background via-background/80 to-transparent"
          aria-hidden
        />
        <div
          className={`relative mx-auto w-full max-w-3xl rounded-[26px] border bg-card/85 shadow-xl shadow-black/5 backdrop-blur-xl transition-all ${
            dragging ? "border-primary/60 ring-2 ring-primary/30" : "border-border/70"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (e.dataTransfer.files?.length) onUpload(e.dataTransfer.files);
          }}
        >
          <ComposerPrimitive.Root className="flex flex-col p-2">
            <ComposerChips />
            <div className="flex items-end gap-1.5">
              <ComposerControls />
              <ComposerPrimitive.Input
                autoFocus
                placeholder="Ask anything, or paste an answer to check…"
                className="max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-2 text-sm leading-relaxed outline-none placeholder:text-muted-foreground/60"
              />
              {isRunning ? (
                <ComposerPrimitive.Cancel
                  title="Stop generating"
                  aria-label="Stop generating"
                  className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-secondary text-foreground transition-colors hover:bg-secondary/70"
                >
                  <Square className="h-3.5 w-3.5 fill-current" aria-hidden />
                </ComposerPrimitive.Cancel>
              ) : (
                <ComposerPrimitive.Send
                  title="Send message"
                  aria-label="Send message"
                  className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-r from-violet-600 to-indigo-600 text-white transition-all hover:from-violet-500 hover:to-indigo-500 disabled:opacity-50"
                >
                  <Send className="h-4 w-4" aria-hidden />
                </ComposerPrimitive.Send>
              )}
            </div>
          </ComposerPrimitive.Root>
        </div>
        <p className="mt-2 flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground">
          <Check className="h-3 w-3" aria-hidden />
          Enter to send · Shift + Enter for a new line
        </p>
      </div>
    </ThreadPrimitive.Root>
  );
}
