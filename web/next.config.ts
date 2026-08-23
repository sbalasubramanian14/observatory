import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
  experimental: {
    // Turbopack's default chunking (see turbopackChunking docs) favours
    // many small, cacheable chunks — good for repeat navigation, bad for
    // a first mobile load: each extra file costs a full round trip under
    // real network latency, and this app's first paint is already gated
    // on all of its JS finishing before it can even start fetching data
    // (spec §4.5 — the bundle is fetched at runtime, never baked in at
    // build time, so there's no way to skip that dependency). Merging
    // aggressively and biasing hard toward the homepage measurably cut
    // first-load JS round trips; see ui-pwa-report.md for the before/after.
    turbopackChunking: {
      minChunkSize: 150000,
      maxChunkCountPerGroup: 8,
      firstPageLoadPriority: 0.9,
      priorityRoutes: [/^\/$/],
      priorityBoost: 2,
    },
  },
};

export default nextConfig;
