import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const runtime = "nodejs";

const FIGURES_DIR = path.resolve(process.cwd(), "..", "artifacts", "figures");

// GET /api/figures/[name] — serves generated experiment figures from artifacts/figures
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ name: string }> }
) {
  const { name } = await params;
  const safe = path.basename(name);
  if (safe !== name || !safe.endsWith(".png")) {
    return new NextResponse("invalid figure name", { status: 400 });
  }
  const filePath = path.join(FIGURES_DIR, safe);
  if (!fs.existsSync(filePath)) {
    return new NextResponse("figure not found — run src/explain/shap_analysis.py", { status: 404 });
  }
  const data = fs.readFileSync(filePath);
  return new NextResponse(data, {
    headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
  });
}
