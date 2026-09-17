"use client";

import { useRemoteThreadListRuntime } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { sqliteThreadListAdapter } from "@/components/chat/sqlite-thread-adapter";

/**
 * Chat runtime with persisted thread history.
 * The outer remote-thread-list runtime owns the SQLite adapter; the inner
 * useChatRuntime detects the nesting and delegates to its AI SDK runtime.
 */
function useThreadChatRuntime() {
  return useChatRuntime();
}

export function useChatThreadsRuntime(options?: {
  threadId?: string;
  onThreadIdChange?: (threadId: string | undefined) => void;
}) {
  return useRemoteThreadListRuntime({
    runtimeHook: useThreadChatRuntime,
    adapter: sqliteThreadListAdapter,
    threadId: options?.threadId,
    onThreadIdChange: options?.onThreadIdChange,
  });
}
