import { chromium } from "playwright";

// Ad-hoc verification for the three owner-reported issues (Issue 1
// truncated summaries, Issue 2 mobile deck density, Issue 3 off-topic
// content). Pass a `phase` name (e.g. "before"/"after") and a base URL.
const PHASE = process.argv[2] || "before";
const BASE = process.argv[3] || "https://the-ai-observatory.vercel.app";
const OUT = "F:/projects/research/web/screenshots";

const browser = await chromium.launch();
let failures = 0;
function check(name, cond) {
  console.log(cond ? `PASS  ${name}` : `FAIL  ${name}`);
  if (!cond) failures++;
}

async function withPage(opts, fn) {
  const ctx = await browser.newContext(opts);
  const page = await ctx.newPage();
  await fn(page, ctx);
  await ctx.close();
}

const VIEWPORTS = [
  { name: "iphone-390x844", width: 390, height: 844 },
  { name: "android-360x800", width: 360, height: 800 },
];

for (const vp of VIEWPORTS) {
  await withPage(
    { viewport: { width: vp.width, height: vp.height }, hasTouch: true, isMobile: true },
    async (page) => {
      await page.goto(BASE + "/", { waitUntil: "networkidle" });
      await page.waitForTimeout(1200);

      // List view first: Issue 1 (truncated summary + read more toggle).
      await page.screenshot({ path: `${OUT}/${PHASE}-${vp.name}-list.png` });
      const readMore = page.getByRole("button", { name: /read more/i }).first();
      const hasReadMore = await readMore.count();
      if (hasReadMore) {
        await readMore.scrollIntoViewIfNeeded();
        await readMore.click();
        await page.waitForTimeout(200);
        await page.screenshot({ path: `${OUT}/${PHASE}-${vp.name}-list-expanded.png` });
      }
      check(`${vp.name}: "Read more" toggle present in list view`, hasReadMore > 0);

      // Cards (deck) view: Issue 2 density.
      const cardsToggle = page.getByLabel(/card view/i);
      if (await cardsToggle.count()) {
        await cardsToggle.click();
        await page.waitForTimeout(700);
        await page.screenshot({ path: `${OUT}/${PHASE}-${vp.name}-cards.png` });

        const deck = page.locator('[class*="StoryDeck-module"][class*="deck"]').first();
        if (await deck.count()) {
          const slideBox = await page.locator('[class*="StoryDeck-module"][class*="slide"]').first().boundingBox();
          const cardBox = await page.locator('[class*="StoryDeck-module"][class*="card"]').first().boundingBox();
          const summaryBox = await page.locator('[class*="StoryDeck-module"][class*="summary"]').first().boundingBox().catch(() => null);
          const headlineBox = await page.locator('[class*="StoryDeck-module"][class*="headline"]').first().boundingBox().catch(() => null);
          console.log(`  ${PHASE}/${vp.name}: slide=${JSON.stringify(slideBox)} card=${JSON.stringify(cardBox)} headline=${JSON.stringify(headlineBox)} summary=${JSON.stringify(summaryBox)}`);
          check(`${vp.name}: card fills most of the viewport height`, cardBox && cardBox.height > vp.height * 0.55);
          check(`${vp.name}: summary text box has visible height`, summaryBox && summaryBox.height > 20);
        }
      }
    },
  );
}

// Issue 3: no off-topic ("World's Fair" film review) content anywhere.
await withPage({ viewport: { width: 1280, height: 900 } }, async (page) => {
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  const bodyText = await page.textContent("body");
  const hasFilmReview = /World.s Fair/i.test(bodyText || "");
  check("desktop feed: no 'World's Fair' film review present", !hasFilmReview);
  await page.screenshot({ path: `${OUT}/${PHASE}-desktop-feed.png`, fullPage: false });
});

await browser.close();
console.log(`\n${PHASE}: ${failures} check(s) failed`);
process.exit(failures > 0 && PHASE === "after" ? 1 : 0);
