"use client";

import { Bot, Send, Sparkles, User } from "lucide-react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";

function UserMessage() {
  return (
    <div className="flex items-end justify-end gap-3">
      <div className="max-w-2xl rounded-2xl rounded-tr-none bg-primary px-4 py-3 text-sm text-primary-foreground">
        <MessagePrimitive.Content />
      </div>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-indigo-500/40 bg-indigo-100 text-indigo-700 dark:border-indigo-500/30 dark:bg-indigo-600/30 dark:text-indigo-300">
        <User className="h-4 w-4" />
      </div>
    </div>
  );
}

function AssistantMessage() {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-violet-500/40 bg-violet-100 text-violet-700 dark:border-violet-500/30 dark:bg-violet-600/30 dark:text-violet-300">
        <Bot className="h-4 w-4" />
      </div>
      <div className="max-w-3xl min-w-0 w-full space-y-3 rounded-2xl rounded-tl-none border border-border bg-secondary/80 px-4 py-3 text-sm">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>
    </div>
  );
}

export function Thread() {
  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col gap-4">
      <ThreadPrimitive.Viewport className="glass-panel flex-1 overflow-y-auto rounded-2xl p-4">
        <ThreadPrimitive.Empty>
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <Sparkles className="h-10 w-10 animate-pulse text-muted-foreground/40" />
            <p className="max-w-sm text-sm text-muted-foreground">
              Ask me to check an answer for hallucination risk. Include the question, a reference
              context, and the candidate answer.
            </p>
            <div className="mt-2 flex max-w-md flex-wrap justify-center gap-2">
              <ThreadPrimitive.Suggestions>
                {({ suggestion }) => (
                  <ThreadPrimitive.Suggestion
                    prompt={suggestion.prompt}
                    send
                    className="rounded-full border border-border bg-secondary/80 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:bg-primary hover:text-primary-foreground"
                  >
                    {suggestion.label}
                  </ThreadPrimitive.Suggestion>
                )}
              </ThreadPrimitive.Suggestions>
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
          placeholder="Paste question, context, or answer to analyze..."
          className="flex-1 bg-transparent px-4 py-2 text-sm outline-none placeholder:text-muted-foreground/60"
        />
        <ComposerPrimitive.Send className="bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground p-2.5 rounded-xl font-semibold transition-all">
          <Send className="h-4 w-4" />
        </ComposerPrimitive.Send>
      </ComposerPrimitive.Root>
    </ThreadPrimitive.Root>
  );
}
