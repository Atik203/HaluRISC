import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const runtime = "nodejs";

const FIGURES_DIR = path.resolve(process.cwd(), "..", "artifacts", "figures");
const ALLOWED_DIRS = new Set(["b3", "b4"]);

// GET /api/figures/[dir]/[name] — B-phase figures (artifacts/figures/b3|b4/*.png)
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ dir: string; name: string }> }
) {
  const { dir, name } = await params;
  if (!ALLOWED_DIRS.has(dir)) {
    return new NextResponse("invalid figure directory", { status: 400 });
  }
  const safe = path.basename(name);
  if (safe !== name || !safe.endsWith(".png")) {
    return new NextResponse("invalid figure name", { status: 400 });
  }
  const filePath = path.join(FIGURES_DIR, dir, safe);
  if (!fs.existsSync(filePath)) {
    return new NextResponse("figure not found", { status: 404 });
  }
  const data = fs.readFileSync(filePath);
  return new NextResponse(data, {
    headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
  });
}
