import { NextResponse } from "next/server";
import { getThread, listMessages, upsertMessage } from "@/lib/chat-db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const format = new URL(request.url).searchParams.get("format") ?? undefined;
  const messages = listMessages(id, format).map((row) => {
    let content: unknown = row.content;
    try {
      content = JSON.parse(row.content);
    } catch {
      /* keep the raw string if a row was written by hand */
    }
    return { id: row.id, parent_id: row.parent_id, format: row.format, content };
  });
  return NextResponse.json({ messages });
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!getThread(id)) {
    return NextResponse.json({ detail: "thread not found" }, { status: 404 });
  }
  let body: { id?: unknown; parent_id?: unknown; format?: unknown; content?: unknown } = {};
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return NextResponse.json({ detail: "invalid JSON body" }, { status: 400 });
  }
  if (typeof body.id !== "string" || !body.id) {
    return NextResponse.json({ detail: "message id required" }, { status: 400 });
  }
  const parentId = typeof body.parent_id === "string" ? body.parent_id : null;
  const format = typeof body.format === "string" && body.format ? body.format : "ai-sdk/v6";
  const content = JSON.stringify(body.content ?? null);
  upsertMessage(id, { id: body.id, parent_id: parentId, format, content });
  return NextResponse.json({ ok: true });
}
