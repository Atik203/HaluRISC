"use client";

import { useMemo, useState, type ReactNode } from "react";
import { createAssistantStream } from "assistant-stream";
import {
  RuntimeAdapterProvider,
  useAui,
  type MessageFormatAdapter,
  type RemoteThreadListAdapter,
  type ThreadHistoryAdapter,
  type ThreadMessage,
} from "@assistant-ui/react";

const API = "/api/chat/threads";

interface ApiThread {
  id: string;
  title: string | null;
  status: string;
  created_at: number;
  updated_at: number;
}

interface ApiMessageRow {
  id: string;
  parent_id: string | null;
  format: string;
  content: unknown;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: init?.body ? { "Content-Type": "application/json", ...(init?.headers ?? {}) } : init?.headers,
  });
  if (!res.ok) throw new Error(`${init?.method ?? "GET"} ${url} failed with ${res.status}`);
  return (await res.json()) as T;
}

function toMetadata(thread: ApiThread) {
  return {
    remoteId: thread.id,
    status: thread.status === "archived" ? ("archived" as const) : ("regular" as const),
    title: thread.title ?? undefined,
    lastMessageAt: new Date(thread.updated_at),
  };
}

/** First user line becomes the sidebar title. */
function titleFromMessages(messages: readonly ThreadMessage[]): string | null {
  for (const message of messages) {
    if (message.role !== "user") continue;
    const text = (message.content as readonly { type?: string; text?: string }[])
      .filter((part) => part.type === "text" && part.text)
      .map((part) => part.text ?? "")
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (text) return text.length > 56 ? `${text.slice(0, 53)}...` : text;
  }
  return null;
}

function createHistoryAdapter(
  getRemoteId: () => string | undefined,
  initialize: () => Promise<string>,
): ThreadHistoryAdapter {
  const persist = async <TMessage, TStorageFormat extends Record<string, unknown>>(
    fmt: MessageFormatAdapter<TMessage, TStorageFormat>,
    item: { parentId: string | null; message: TMessage },
    remoteId: string,
  ) => {
    await request(`${API}/${remoteId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        id: fmt.getId(item.message),
        parent_id: item.parentId,
        format: fmt.format,
        content: fmt.encode(item),
      }),
    });
  };

  return {
    // The plain load/append pair is unused on the AI SDK path; withFormat covers it.
    async load() {
      return { messages: [] };
    },
    async append() {
      /* handled by withFormat */
    },
    withFormat<TMessage, TStorageFormat extends Record<string, unknown>>(
      fmt: MessageFormatAdapter<TMessage, TStorageFormat>,
    ) {
      return {
        async load() {
          const remoteId = getRemoteId();
          if (!remoteId) return { messages: [] };
          const { messages } = await request<{ messages: ApiMessageRow[] }>(
            `${API}/${remoteId}/messages?format=${encodeURIComponent(fmt.format)}`,
          );
          return {
            messages: messages.map((row) =>
              fmt.decode({
                id: row.id,
                parent_id: row.parent_id,
                format: row.format,
                content: row.content as TStorageFormat,
              }),
            ),
          };
        },
        async append(item: { parentId: string | null; message: TMessage }) {
          const remoteId = await initialize();
          await persist(fmt, item, remoteId);
        },
        async update(item: { parentId: string | null; message: TMessage }) {
          const remoteId = getRemoteId();
          if (!remoteId) return;
          await persist(fmt, item, remoteId);
        },
      };
    },
  };
}

/** Supplies the SQLite-backed history adapter to every thread runtime. */
function HistoryProvider({ children }: { children?: ReactNode }) {
  const aui = useAui();
  const [history] = useState(() =>
    createHistoryAdapter(
      () => aui.threadListItem.getState().remoteId,
      async () => {
        const { remoteId } = await aui.threadListItem.initialize();
        return remoteId;
      },
    ),
  );
  const adapters = useMemo(() => ({ history }), [history]);
  return <RuntimeAdapterProvider adapters={adapters}>{children}</RuntimeAdapterProvider>;
}

/** assistant-ui remote thread list adapter backed by the local SQLite store. */
export const sqliteThreadListAdapter: RemoteThreadListAdapter = {
  unstable_Provider: HistoryProvider,

  async list() {
    const { threads } = await request<{ threads: ApiThread[] }>(API);
    return { threads: threads.map(toMetadata) };
  },

  async initialize(threadId) {
    const { remoteId } = await request<{ remoteId: string }>(API, {
      method: "POST",
      body: JSON.stringify({ id: threadId }),
    });
    return { remoteId };
  },

  async fetch(threadId) {
    const { thread } = await request<{ thread: ApiThread }>(`${API}/${threadId}`);
    return toMetadata(thread);
  },

  async rename(remoteId, newTitle) {
    await request(`${API}/${remoteId}`, { method: "PATCH", body: JSON.stringify({ title: newTitle }) });
  },

  async updateCustom(remoteId, custom) {
    await request(`${API}/${remoteId}`, { method: "PATCH", body: JSON.stringify({ custom: custom ?? null }) });
  },

  async archive(remoteId) {
    await request(`${API}/${remoteId}`, { method: "PATCH", body: JSON.stringify({ status: "archived" }) });
  },

  async unarchive(remoteId) {
    await request(`${API}/${remoteId}`, { method: "PATCH", body: JSON.stringify({ status: "regular" }) });
  },

  async delete(remoteId) {
    await request(`${API}/${remoteId}`, { method: "DELETE" });
  },

  async generateTitle(remoteId, messages) {
    const title = titleFromMessages(messages) ?? "New chat";
    try {
      await request(`${API}/${remoteId}`, { method: "PATCH", body: JSON.stringify({ title }) });
    } catch {
      /* title is cosmetic; keep the thread usable if the rename fails */
    }
    return createAssistantStream((controller) => {
      controller.appendText(title);
    });
  },
};
