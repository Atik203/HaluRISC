import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const runtime = "nodejs";

const FIGURES_DIR = path.resolve(process.cwd(), "..", "artifacts", "figures");
const ALLOWED_DIRS = new Set(["b3", "b4"]);

// GET /api/figures/[...path] — serves generated figures:
//   /api/figures/fig_reliability.png        (top-level)
//   /api/figures/b3/context_length.png      (B-phase subfolders, whitelisted)
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await params;
  if (segments.length === 0 || segments.length > 2) {
    return new NextResponse("invalid figure path", { status: 400 });
  }
  const file = segments[segments.length - 1];
  const safe = path.basename(file);
  if (safe !== file || !safe.endsWith(".png")) {
    return new NextResponse("invalid figure name", { status: 400 });
  }
  let rel: string;
  if (segments.length === 1) {
    rel = safe;
  } else {
    const dir = segments[0];
    if (!ALLOWED_DIRS.has(dir)) {
      return new NextResponse("invalid figure directory", { status: 400 });
    }
    rel = path.join(dir, safe);
  }
  const filePath = path.join(FIGURES_DIR, rel);
  if (!fs.existsSync(filePath)) {
    return new NextResponse("figure not found", { status: 404 });
  }
  const data = fs.readFileSync(filePath);
  return new NextResponse(data, {
    headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
  });
}
