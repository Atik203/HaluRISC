import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

/**
 * Server-only SQLite store for chat history.
 * The database file is created on first use at web/data/chat.sqlite (gitignored).
 * Schema mirrors the assistant-ui persistence contract:
 *   threads  -> remoteId, title, status, custom
 *   messages -> (id, parent_id, format, content) per thread
 */

const DB_DIR = path.join(process.cwd(), "data");
const DB_PATH = path.join(DB_DIR, "chat.sqlite");

export interface ThreadRecord {
  id: string;
  title: string | null;
  status: string;
  custom: string | null;
  created_at: number;
  updated_at: number;
}

export interface MessageRecord {
  seq: number;
  thread_id: string;
  id: string;
  parent_id: string | null;
  format: string;
  content: string;
}

interface GlobalWithDb {
  __haluriscChatDb?: DatabaseSync;
}

function openDb(): DatabaseSync {
  fs.mkdirSync(DB_DIR, { recursive: true });
  const db = new DatabaseSync(DB_PATH);
  db.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS threads (
      id TEXT PRIMARY KEY,
      title TEXT,
      status TEXT NOT NULL DEFAULT 'regular',
      custom TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      thread_id TEXT NOT NULL,
      id TEXT NOT NULL,
      parent_id TEXT,
      format TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      UNIQUE (thread_id, id)
    );
    CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages (thread_id, seq);
  `);
  return db;
}

export function getDb(): DatabaseSync {
  const g = globalThis as unknown as GlobalWithDb;
  if (!g.__haluriscChatDb) g.__haluriscChatDb = openDb();
  return g.__haluriscChatDb;
}

/* ------------------------------------------------------------------ */
/* Threads                                                             */
/* ------------------------------------------------------------------ */

export function listThreads(): ThreadRecord[] {
  return getDb()
    .prepare("SELECT * FROM threads ORDER BY updated_at DESC, created_at DESC")
    .all() as unknown as ThreadRecord[];
}

export function getThread(id: string): ThreadRecord | null {
  const row = getDb().prepare("SELECT * FROM threads WHERE id = ?").get(id);
  return (row as unknown as ThreadRecord | undefined) ?? null;
}

export function createThread(id: string, title: string | null = null): ThreadRecord {
  const now = Date.now();
  getDb()
    .prepare("INSERT OR IGNORE INTO threads (id, title, status, custom, created_at, updated_at) VALUES (?, ?, 'regular', NULL, ?, ?)")
    .run(id, title, now, now);
  const row = getThread(id);
  if (!row) throw new Error(`thread ${id} could not be created`);
  return row;
}

export function updateThread(
  id: string,
  patch: { title?: string; status?: string; custom?: string | null },
): ThreadRecord | null {
  const row = getThread(id);
  if (!row) return null;
  const next = {
    title: patch.title !== undefined ? patch.title : row.title,
    status: patch.status !== undefined ? patch.status : row.status,
    custom: patch.custom !== undefined ? patch.custom : row.custom,
  };
  getDb()
    .prepare("UPDATE threads SET title = ?, status = ?, custom = ?, updated_at = ? WHERE id = ?")
    .run(next.title, next.status, next.custom, Date.now(), id);
  return getThread(id);
}

export function deleteThread(id: string): boolean {
  const db = getDb();
  db.prepare("DELETE FROM messages WHERE thread_id = ?").run(id);
  const res = db.prepare("DELETE FROM threads WHERE id = ?").run(id);
  return Number(res.changes) > 0;
}

/* ------------------------------------------------------------------ */
/* Messages                                                            */
/* ------------------------------------------------------------------ */

export function listMessages(threadId: string, format?: string): MessageRecord[] {
  if (format) {
    return getDb()
      .prepare("SELECT * FROM messages WHERE thread_id = ? AND format = ? ORDER BY seq ASC")
      .all(threadId, format) as unknown as MessageRecord[];
  }
  return getDb()
    .prepare("SELECT * FROM messages WHERE thread_id = ? ORDER BY seq ASC")
    .all(threadId) as unknown as MessageRecord[];
}

/** Derive a sidebar title from the first user message payload. */
function deriveTitle(contentJson: string): string | null {
  try {
    const parsed = JSON.parse(contentJson) as {
      role?: string;
      parts?: Array<{ type?: string; text?: string }>;
    };
    if (parsed.role !== "user") return null;
    const text = (parsed.parts ?? [])
      .filter((p) => p.type === "text" && typeof p.text === "string")
      .map((p) => p.text ?? "")
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) return null;
    return text.length > 56 ? `${text.slice(0, 53)}...` : text;
  } catch {
    return null;
  }
}

export function upsertMessage(
  threadId: string,
  row: { id: string; parent_id: string | null; format: string; content: string },
): void {
  const db = getDb();
  const now = Date.now();
  // A new thread may be persisted before its row exists (first-message race).
  createThread(threadId);
  db.prepare(
    `INSERT INTO messages (thread_id, id, parent_id, format, content, created_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT (thread_id, id) DO UPDATE SET parent_id = excluded.parent_id, format = excluded.format, content = excluded.content`,
  ).run(threadId, row.id, row.parent_id, row.format, row.content, now);

  const thread = getThread(threadId);
  const title = !thread?.title ? deriveTitle(row.content) : null;
  db.prepare("UPDATE threads SET title = COALESCE(title, ?), updated_at = ? WHERE id = ?").run(
    title,
    now,
    threadId,
  );
}
