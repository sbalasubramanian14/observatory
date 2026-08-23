// One-off icon generator for the PWA manifest's icon set. Rasterizes the
// Observatory mark (a dark aperture with an accent ring/orbit + a small
// "star" point — same motif as src/app/icon.svg, redrawn here so the
// maskable variant can bake in its own safe-zone padding) via a headless
// Chromium page, since this repo has no image library installed. Run with:
//   node scripts/generate-icons.mjs
// Not part of the build; output is committed to public/icons/.
import { chromium } from "playwright";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = join(__dirname, "..", "public", "icons");
mkdirSync(outDir, { recursive: true });

// scale: fraction of the viewBox the mark occupies (smaller = more padding,
// needed for maskable icons so the mark survives an OS circular crop).
function markSvg({ size, scale, bg }) {
  const c = size / 2;
  const r = (size / 2) * scale;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <rect width="${size}" height="${size}" fill="${bg}"/>
  <circle cx="${c}" cy="${c}" r="${r * 0.66}" stroke="#8b9dff" stroke-width="${Math.max(2, size * 0.028)}" fill="none"/>
  <circle cx="${c}" cy="${c}" r="${r * 0.24}" fill="#8b9dff"/>
  <circle cx="${c + r * 0.62}" cy="${c - r * 0.62}" r="${r * 0.15}" fill="#f2916f"/>
</svg>`;
}

const targets = [
  { file: "icon-192.png", size: 192, scale: 0.82, bg: "#131319", purpose: "any" },
  { file: "icon-512.png", size: 512, scale: 0.82, bg: "#131319", purpose: "any" },
  { file: "icon-maskable-192.png", size: 192, scale: 0.5, bg: "#131319", purpose: "maskable" },
  { file: "icon-maskable-512.png", size: 512, scale: 0.5, bg: "#131319", purpose: "maskable" },
  { file: "apple-touch-icon.png", size: 180, scale: 0.78, bg: "#131319", purpose: "any" },
  { file: "favicon-32.png", size: 32, scale: 0.86, bg: "#131319", purpose: "any" },
  { file: "favicon-16.png", size: 16, scale: 0.86, bg: "#131319", purpose: "any" },
];

const browser = await chromium.launch();
const page = await browser.newPage();

for (const t of targets) {
  const svg = markSvg(t);
  await page.setViewportSize({ width: t.size, height: t.size });
  await page.setContent(
    `<!doctype html><html><head><style>html,body{margin:0;padding:0;}</style></head><body>${svg}</body></html>`,
  );
  const el = await page.$("svg");
  const buf = await el.screenshot({ omitBackground: false });
  writeFileSync(join(outDir, t.file), buf);
  console.log("wrote", t.file);
}

await browser.close();
