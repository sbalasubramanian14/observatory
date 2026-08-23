import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:5000";
const browser = await chromium.launch();

async function measure(label, { reducedMotion, cpuThrottle }) {
  const ctx = await browser.newContext({
    viewport: { width: 390, height: 844 },
    reducedMotion: reducedMotion ? "reduce" : "no-preference",
  });
  const page = await ctx.newPage();
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Performance.enable");
  if (cpuThrottle) await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpuThrottle });

  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);

  // Sample raw animation-frame deltas for 3s while idle (characterizes the
  // shader loop's own steady-state cost, isolated from scroll/layout work).
  const idleFrames = await page.evaluate(() => {
    return new Promise((resolve) => {
      const deltas = [];
      let last = performance.now();
      function tick(now) {
        deltas.push(now - last);
        last = now;
        if (deltas.length < 150) requestAnimationFrame(tick);
        else resolve(deltas);
      }
      requestAnimationFrame(tick);
    });
  });

  const before = await cdp.send("Performance.getMetrics");

  // Scroll through the feed for 3s to stress layout/paint concurrently
  // with the shader loop -- this is the actual "does it jank scrolling"
  // question, not just idle GPU draw cost.
  const scrollStart = Date.now();
  const frameTimes = [];
  while (Date.now() - scrollStart < 3000) {
    const t0 = Date.now();
    await page.mouse.wheel(0, 120);
    await page.waitForTimeout(16);
    frameTimes.push(Date.now() - t0);
  }

  const after = await cdp.send("Performance.getMetrics");

  function metric(list, name) {
    return list.metrics.find((m) => m.name === name)?.value ?? 0;
  }
  const taskDuration = metric(after, "TaskDuration") - metric(before, "TaskDuration");
  const scriptDuration = metric(after, "ScriptDuration") - metric(before, "ScriptDuration");
  const layoutDuration = metric(after, "LayoutDuration") - metric(before, "LayoutDuration");
  const wallSeconds = (metric(after, "Timestamp") - metric(before, "Timestamp"));

  const avgIdleFrame = idleFrames.reduce((a, b) => a + b, 0) / idleFrames.length;
  const maxIdleFrame = Math.max(...idleFrames);
  const droppedScrollFrames = frameTimes.filter((t) => t > 50).length;

  console.log(`\n=== ${label} (cpuThrottle=${cpuThrottle || 1}x, reducedMotion=${reducedMotion}) ===`);
  console.log(`idle: avg frame ${avgIdleFrame.toFixed(2)}ms, max frame ${maxIdleFrame.toFixed(2)}ms (target 33ms @ 30fps cap)`);
  console.log(`scroll (3s window): main-thread TaskDuration ${taskDuration.toFixed(3)}s / wall ${wallSeconds.toFixed(3)}s = ${((taskDuration / wallSeconds) * 100).toFixed(1)}% busy`);
  console.log(`  script ${scriptDuration.toFixed(3)}s, layout ${layoutDuration.toFixed(3)}s`);
  console.log(`  dropped scroll frames (>50ms): ${droppedScrollFrames} / ${frameTimes.length}`);

  await ctx.close();
  return { avgIdleFrame, maxIdleFrame, taskDuration, wallSeconds, droppedScrollFrames };
}

await measure("Shader ON, no throttle", { reducedMotion: false, cpuThrottle: 1 });
const shaderThrottled = await measure("Shader ON, 4x CPU throttle (simulated mid-tier)", { reducedMotion: false, cpuThrottle: 4 });
const fallbackThrottled = await measure("CSS fallback (reduced-motion), 4x CPU throttle", { reducedMotion: true, cpuThrottle: 4 });

console.log("\n=== SUMMARY ===");
console.log(`Shader busy% at 4x throttle: ${((shaderThrottled.taskDuration / shaderThrottled.wallSeconds) * 100).toFixed(1)}%`);
console.log(`Fallback busy% at 4x throttle: ${((fallbackThrottled.taskDuration / fallbackThrottled.wallSeconds) * 100).toFixed(1)}%`);
console.log(`Incremental cost of shader vs fallback: ${(((shaderThrottled.taskDuration - fallbackThrottled.taskDuration) / shaderThrottled.wallSeconds) * 100).toFixed(1)} percentage points`);
console.log(`Shader dropped frames at 4x throttle: ${shaderThrottled.droppedScrollFrames}, fallback: ${fallbackThrottled.droppedScrollFrames}`);

await browser.close();
