"use client";

import { useEffect, useRef, useState } from "react";
import { useAui, useAuiState } from "@assistant-ui/react";
import { Check, MessageSquarePlus, Pencil, Trash2, X } from "lucide-react";
import { toast } from "@/components/ui/toast";

function formatWhen(date?: Date): string {
  if (!date) return "";
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function ThreadSidebar({
  className = "",
  onNavigate,
}: {
  className?: string;
  onNavigate?: () => void;
}) {
  const aui = useAui();
  const items = useAuiState((s) => s.threads.threadItems);
  const mainId = useAuiState((s) => s.threads.mainThreadId);
  const isLoading = useAuiState((s) => s.threads.isLoading);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  const visibleItems = items.filter((item) => item.status !== "archived");

  const startRename = (id: string, title: string | undefined) => {
    setConfirmId(null);
    setEditingId(id);
    setDraft(title ?? "");
  };

  const commitRename = (id: string) => {
    const next = draft.trim();
    setEditingId(null);
    if (next) {
      aui.threads.item({ id }).rename(next);
      toast("Chat renamed");
    }
  };

  const remove = (id: string) => {
    setConfirmId(null);
    aui.threads.item({ id }).delete();
    if (id === mainId) aui.threads.switchToNewThread();
    onNavigate?.();
    toast("Chat deleted");
  };

  return (
    <aside
      className={`glass-panel flex min-h-0 flex-col rounded-2xl ${className}`}
      aria-label="Chat history"
    >
      <div className="flex items-center justify-between gap-2 border-b border-border/40 px-3 py-2.5">
        <span className="eyebrow">Chats</span>
        <button
          type="button"
          onClick={() => {
            aui.threads.switchToNewThread();
            onNavigate?.();
          }}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border/70 bg-secondary/50 px-2.5 py-1.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" aria-hidden />
          New chat
        </button>
      </div>

      <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2">
        {isLoading && visibleItems.length === 0 && (
          <div className="space-y-1.5 p-1" aria-hidden>
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-11 rounded-xl bg-secondary/50 animate-pulse-soft" />
            ))}
          </div>
        )}

        {!isLoading && visibleItems.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] leading-relaxed text-muted-foreground">
            No saved chats yet. Ask something and it will appear here.
          </p>
        )}

        {visibleItems.map((item) => {
          const active = item.id === mainId;
          const editing = editingId === item.id;
          const confirming = confirmId === item.id;
          const title = item.title?.trim() ? item.title : "New chat";
          return (
            <div
              key={item.id}
              className={`group rounded-xl border transition-colors ${
                active
                  ? "border-primary/40 bg-primary/10"
                  : "border-transparent hover:border-border/70 hover:bg-secondary/40"
              }`}
            >
              {editing ? (
                <div className="flex items-center gap-1 p-1.5">
                  <input
                    ref={inputRef}
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(item.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    aria-label="Chat title"
                    className="min-w-0 flex-1 rounded-lg border border-border bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <button
                    type="button"
                    onClick={() => commitRename(item.id)}
                    aria-label="Save title"
                    className="rounded-md p-1 text-emerald-600 hover:bg-secondary dark:text-emerald-400"
                  >
                    <Check className="h-3.5 w-3.5" aria-hidden />
                  </button>
                  <button
                    type="button"
                    onClick={() => setEditingId(null)}
                    aria-label="Cancel rename"
                    className="rounded-md p-1 text-muted-foreground hover:bg-secondary"
                  >
                    <X className="h-3.5 w-3.5" aria-hidden />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-1 p-1.5">
                  <button
                    type="button"
                    onClick={() => {
                      if (!active) aui.threads.switchToThread(item.id);
                      onNavigate?.();
                    }}
                    aria-current={active ? "true" : undefined}
                    className="min-w-0 flex-1 rounded-lg px-2 py-1.5 text-left"
                  >
                    <span className="block truncate text-xs font-semibold">{title}</span>
                    {item.lastMessageAt && (
                      <span className="mt-0.5 block font-mono text-[10px] text-muted-foreground">
                        {formatWhen(item.lastMessageAt)}
                      </span>
                    )}
                  </button>

                  <span className="flex shrink-0 items-center gap-0.5 opacity-100 transition-opacity sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100">
                    {confirming ? (
                      <>
                        <button
                          type="button"
                          onClick={() => remove(item.id)}
                          aria-label={`Confirm delete ${title}`}
                          className="rounded-md px-1.5 py-1 text-[10px] font-bold text-rose-600 hover:bg-rose-500/10 dark:text-rose-400"
                        >
                          Delete
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirmId(null)}
                          aria-label="Cancel delete"
                          className="rounded-md p-1 text-muted-foreground hover:bg-secondary"
                        >
                          <X className="h-3.5 w-3.5" aria-hidden />
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => startRename(item.id, item.title)}
                          aria-label={`Rename ${title}`}
                          className="rounded-md p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
                        >
                          <Pencil className="h-3.5 w-3.5" aria-hidden />
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirmId(item.id)}
                          aria-label={`Delete ${title}`}
                          className="rounded-md p-1 text-muted-foreground hover:bg-rose-500/10 hover:text-rose-600 dark:hover:text-rose-400"
                        >
                          <Trash2 className="h-3.5 w-3.5" aria-hidden />
                        </button>
                      </>
                    )}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </nav>

      <p className="border-t border-border/40 px-3 py-2 text-[10px] leading-relaxed text-muted-foreground">
        Saved in a local SQLite file on this machine.
      </p>
    </aside>
  );
}
