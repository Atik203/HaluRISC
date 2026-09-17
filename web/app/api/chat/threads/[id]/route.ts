import { NextResponse } from "next/server";
import { deleteThread, getThread, updateThread } from "@/lib/chat-db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const thread = getThread(id);
  if (!thread) return NextResponse.json({ detail: "thread not found" }, { status: 404 });
  return NextResponse.json({ thread });
}

export async function PATCH(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let body: { title?: unknown; status?: unknown; custom?: unknown } = {};
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return NextResponse.json({ detail: "invalid JSON body" }, { status: 400 });
  }
  const patch: { title?: string; status?: string; custom?: string | null } = {};
  if (typeof body.title === "string") patch.title = body.title;
  if (typeof body.status === "string") patch.status = body.status === "archived" ? "archived" : "regular";
  if (body.custom === null) patch.custom = null;
  else if (body.custom !== undefined) patch.custom = JSON.stringify(body.custom);

  const thread = updateThread(id, patch);
  if (!thread) return NextResponse.json({ detail: "thread not found" }, { status: 404 });
  return NextResponse.json({ thread });
}

export async function DELETE(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const removed = deleteThread(id);
  if (!removed) return NextResponse.json({ detail: "thread not found" }, { status: 404 });
  return NextResponse.json({ ok: true });
}
