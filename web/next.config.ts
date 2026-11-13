import { withAui } from "@assistant-ui/next";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Frontend-only deploys (e.g. Vercel) have no FastAPI backend: skip the
    // /api/ml proxy so those paths 404 fast and the UI fallbacks engage.
    // Local dev keeps the localhost proxy unless ML_API_URL overrides it.
    const mlTarget = process.env.ML_API_URL ?? (process.env.VERCEL ? undefined : "http://127.0.0.1:8000");
    if (!mlTarget) return [];
    return [
      {
        source: "/api/ml/:path*",
        destination: `${mlTarget}/:path*`,
      },
    ];
  },
};

export default withAui(nextConfig);
