#!/usr/bin/env node
// Copies the published bundle (F:/projects/research/public) into
// web/public/data so the dev server and static export can serve it at
// relative URLs (/data/...). This mirrors what a CDN would serve in
// production — see src/lib/bundle.ts and spec §4.1/§4.5.
import { cp, rm, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const source = path.join(repoRoot, "public");
const dest = path.join(__dirname, "..", "public", "data");

if (!existsSync(source)) {
  console.error(`Bundle source not found at ${source}`);
  process.exit(1);
}

await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });
await cp(source, dest, { recursive: true });

console.log(`Synced bundle: ${source} -> ${dest}`);
