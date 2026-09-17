import { NextResponse } from "next/server";
import { createThread, listThreads } from "@/lib/chat-db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ threads: listThreads() });
}

export async function POST(request: Request) {
  let body: { id?: unknown; title?: unknown } = {};
  try {
    body = (await request.json()) as typeof body;
  } catch {
    /* empty body allowed */
  }
  const id = typeof body.id === "string" && body.id.trim() ? body.id.trim() : crypto.randomUUID();
  const title = typeof body.title === "string" && body.title.trim() ? body.title.trim() : null;
  const thread = createThread(id, title);
  return NextResponse.json({ remoteId: thread.id, thread });
}
