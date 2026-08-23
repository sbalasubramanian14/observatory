import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:5000";
const OUT = "F:/projects/research/web/screenshots";
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
const page = await ctx.newPage();

console.log("1. First visit online — populate SW caches + lastSync...");
await page.goto(BASE + "/", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);

// wait for SW to actually be activated AND controlling *this* page --
// reaching "activated" state and having claimed this already-open client
// are two different moments (clients.claim() runs inside activate, but
// the browser can lag marking navigator.serviceWorker.controller).
const swState = await page.evaluate(async () => {
  if (!("serviceWorker" in navigator)) return "unsupported";
  const reg = await navigator.serviceWorker.ready;
  const deadline = Date.now() + 5000;
  while (!navigator.serviceWorker.controller && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 100));
  }
  return {
    active: reg.active ? reg.active.state : "no-active-worker",
    controlling: !!navigator.serviceWorker.controller,
  };
});
console.log("service worker state:", swState);

// visit a story detail too, so its content is cached
const link = page.locator("a[href^='/story/']").first();
const href = await link.getAttribute("href");
await link.click();
await page.waitForTimeout(1200);
console.log("visited story:", href);

// sw.js defers precaching the non-critical route shells (story/saved/
// dismissed/sources) by ~4s so that work never competes with this same
// page's own first-load fetches for bandwidth (see sw.js's precacheDeferred
// comment) -- a realistic "one visit" session (read the feed, open a
// story) comfortably outlasts that delay, so wait for it here too rather
// than testing an artificially fast go-offline.
await page.waitForTimeout(4000);

// check cache contents
const cacheInfo = await page.evaluate(async () => {
  const names = await caches.keys();
  const counts = {};
  for (const name of names) {
    const cache = await caches.open(name);
    const keys = await cache.keys();
    counts[name] = keys.length;
  }
  return counts;
});
console.log("caches populated:", cacheInfo);

const lastSyncBefore = await page.evaluate(() => localStorage.getItem("observatory:lastSync"));
console.log("lastSync recorded:", lastSyncBefore, lastSyncBefore ? new Date(Number(lastSyncBefore)).toISOString() : null);

console.log("\n2. Going offline and reloading feed...");
await ctx.setOffline(true);
await page.goto(BASE + "/", { waitUntil: "load", timeout: 15000 }).catch((e) => console.log("goto error:", e.message));
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/redesign-offline-feed.png` });

const bodyText = await page.locator("body").innerText();
const hasOfflineBanner = bodyText.includes("offline") || bodyText.includes("Offline");
console.log("offline banner present:", hasOfflineBanner);
const storiesVisible = await page.locator("article").count();
console.log("stories rendered while offline:", storiesVisible);

console.log("\n3. Going offline and reloading the previously-visited story detail...");
await page.goto(BASE + href, { waitUntil: "load", timeout: 15000 }).catch((e) => console.log("goto error:", e.message));
await page.waitForTimeout(1200);
await page.screenshot({ path: `${OUT}/redesign-offline-story-detail.png` });
const storyBodyText = await page.locator("body").innerText();
console.log("story detail rendered offline, has title text:", storyBodyText.length > 200);

await ctx.setOffline(false);
await browser.close();
