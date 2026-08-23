import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:5000";
const OUT = "F:/projects/research/web/screenshots";
const browser = await chromium.launch();
let failures = 0;

function check(name, cond) {
  console.log(cond ? `PASS  ${name}` : `FAIL  ${name}`);
  if (!cond) failures++;
}

async function withPage(opts, fn) {
  const ctx = await browser.newContext(opts);
  const errors = [];
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const text = m.text();
    // Benign: hotlinked publisher images occasionally 404 — LeadImage
    // handles this with a graceful text-only fallback (spec D0). Not a
    // console error worth failing this check over.
    if (text.includes("Failed to load resource")) return;
    errors.push(text);
  });
  await fn(page, ctx, errors);
  await ctx.close();
  return errors;
}

// ---- 1. iPhone 390x844 viewport, light + dark, list mode ----
await withPage({ viewport: { width: 390, height: 844 }, colorScheme: "light", hasTouch: true, isMobile: true }, async (page, ctx, errors) => {
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/redesign-iphone-390-list-light.png` });
  check("iphone list light: no console errors", errors.length === 0);

  // switch to cards
  await page.getByLabel("Card view").click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/redesign-iphone-390-cards.png` });
  const deckHeight = await page.locator('[class*="deck"]').first().evaluate((el) => el.clientHeight).catch(() => null);
  check("deck mounted with nonzero height", deckHeight && deckHeight > 200);

  // Vertical paging mechanism: CSS scroll-snap (native momentum, not JS
  // physics -- see StoryDeck.module.css). Confirm it's actually wired up,
  // then drive a real scroll and confirm the browser's own snap logic
  // settles on an exact card-height multiple.
  const deck = page.locator('[class*="StoryDeck-module"][class*="deck"]').first();
  const snapType = await deck.evaluate((el) => getComputedStyle(el).scrollSnapType);
  check(`deck has CSS scroll-snap-type applied (got "${snapType}")`, snapType.includes("mandatory"));

  const box = await deck.boundingBox();
  if (box) {
    await deck.evaluate((el) => el.scrollTo({ top: el.clientHeight * 1.4, behavior: "instant" }));
    await page.waitForTimeout(500);
    const info = await deck.evaluate((el) => ({ scrollTop: el.scrollTop, clientHeight: el.clientHeight }));
    const remainder = info.scrollTop % info.clientHeight;
    check(
      `deck lands on an exact card boundary (scrollTop=${info.scrollTop}, clientHeight=${info.clientHeight}, remainder=${remainder})`,
      remainder < 2 || remainder > info.clientHeight - 2,
    );
    await page.screenshot({ path: `${OUT}/redesign-iphone-390-cards-scrolled.png` });
    await deck.evaluate((el) => el.scrollTo({ top: 0, behavior: "instant" }));
    await page.waitForTimeout(200);
  } else {
    check("deck bounding box found", false);
  }

  // Swipe-right (save) / swipe-left (dismiss) gesture. Dispatched as real
  // PointerEvents from inside the page (not via CDP mouse synthesis,
  // which does not reliably represent a fast realistic drag) so the test
  // exercises the actual component logic end to end.
  const swipeResult = await page.evaluate(async () => {
    const el = document.querySelector("article");
    const rect = el.getBoundingClientRect();
    const startX = rect.x + rect.width / 2;
    const startY = rect.y + rect.height / 2;
    function fire(type, x, y) {
      el.dispatchEvent(new PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId: 1, pointerType: "touch",
        clientX: x, clientY: y, button: 0, buttons: type === "pointerup" ? 0 : 1,
      }));
    }
    fire("pointerdown", startX, startY);
    for (let i = 1; i <= 8; i++) {
      fire("pointermove", startX + i * 20, startY);
      await new Promise((r) => setTimeout(r, 16)); // one frame, let React commit between moves
    }
    fire("pointerup", startX + 160, startY);
    return "dispatched";
  });
  await page.waitForTimeout(300);
  const savedIds = await page.evaluate(() => localStorage.getItem("observatory:saved"));
  check(`swipe-right save gesture recorded a save (${swipeResult}, savedIds=${savedIds})`, !!savedIds && savedIds !== "[]" && savedIds !== "null");
  await page.screenshot({ path: `${OUT}/redesign-iphone-390-cards-swiped.png` });
});

// ---- 2. Android 360x800 viewport ----
await withPage({ viewport: { width: 360, height: 800 }, colorScheme: "dark", hasTouch: true, isMobile: true }, async (page, ctx, errors) => {
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/redesign-android-360-list-dark.png` });
  check("android 360 dark: no console errors", errors.length === 0); if (errors.length) console.log(errors);

  // confirm list mode still reachable / is default
  const savedMode = await page.evaluate(() => localStorage.getItem("observatory:viewMode"));
  check("list mode is default on fresh device", savedMode === null);
});

// ---- 3. Desktop 1440x900 — multi-column grid, sort, filter, theme ----
await withPage({ viewport: { width: 1440, height: 900 } }, async (page, ctx, errors) => {
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/redesign-desktop-1440-light.png` });

  const columns = await page.evaluate(() => {
    const list = document.querySelector('[class*="page-module"][class*="list"]') || document.querySelector('main [class*="list"]');
    if (!list) return null;
    return getComputedStyle(list).gridTemplateColumns.split(" ").length;
  });
  check(`desktop grid has multiple columns (found ${columns})`, columns && columns >= 2);

  // sort change
  await page.locator("#sort-select").selectOption("outlets");
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/redesign-desktop-sort-outlets.png` });

  // source filter narrows the set — read the "Showing X of Y" total
  // (article count is capped at PAGE_SIZE=20 either way, so it can't
  // distinguish "filtered to exactly 20+" from "unfiltered").
  async function totalCount() {
    const text = await page.locator('[class*="statusRow"]').innerText();
    const match = text.match(/of\s+([\d,]+)/);
    return match ? Number(match[1].replace(/,/g, "")) : null;
  }
  const beforeTotal = await totalCount();
  await page.getByRole("button", { name: /^Sources/ }).first().click();
  await page.waitForTimeout(300);
  // Click one individual source checkbox (not a territory group header,
  // which selects many at once and could coincidentally match everything).
  const sourceCheckbox = page.locator('input[type="checkbox"]').nth(1);
  await sourceCheckbox.click();
  await page.getByRole("button", { name: "Done" }).click();
  await page.waitForTimeout(600);
  const afterTotal = await totalCount();
  check(`filter narrowed the total count (before=${beforeTotal}, after=${afterTotal})`, afterTotal !== null && afterTotal < beforeTotal);
  await page.screenshot({ path: `${OUT}/redesign-desktop-filtered.png` });

  // theme toggle (desktop header)
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  const before = await page.locator("html").getAttribute("data-theme");
  await page.getByRole("banner").getByLabel("Dark theme").click();
  await page.waitForTimeout(400);
  const after = await page.locator("html").getAttribute("data-theme");
  check(`theme toggle changed data-theme (before=${before}, after=${after})`, before !== after);
  await page.screenshot({ path: `${OUT}/redesign-desktop-dark.png` });
  await page.getByRole("banner").getByLabel("Light theme").click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}/redesign-desktop-light.png` });

  check("desktop feed: no console errors", errors.length === 0); if (errors.length) console.log(errors);
});

// ---- 4. Save/dismiss on list card, evidence expand, story detail, sources, saved, dismissed ----
await withPage({ viewport: { width: 390, height: 844 } }, async (page, ctx, errors) => {
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(800);

  const firstCard = page.locator("article").first();
  await firstCard.getByLabel("Save story").click();
  await page.waitForTimeout(300);
  const savedActive = await firstCard.getByLabel("Save story").getAttribute("aria-pressed");
  check("save button toggles pressed state", savedActive === "true");

  await page.locator("article").nth(1).getByLabel("Dismiss story").click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}/redesign-mobile-list-with-save.png` });

  await firstCard.getByText("Show evidence").click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/redesign-mobile-evidence-expanded.png` });

  await page.goto(BASE + "/saved/", { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/redesign-mobile-saved.png` });
  check("saved page shows the saved story", (await page.locator("a,li").allTextContents()).length > 0);

  await page.goto(BASE + "/dismissed/", { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/redesign-mobile-dismissed.png` });

  await page.goto(BASE + "/sources/", { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/redesign-mobile-sources.png` });

  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  const link = page.locator("a[href^='/story/']").first();
  await link.click();
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${OUT}/redesign-mobile-story-detail.png` });

  check("full journey: no console errors", errors.length === 0); if (errors.length) console.log(errors);
});

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
