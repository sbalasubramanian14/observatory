"use client";

// The PWA redesign's signature element (see docs/superpowers/sdd for the
// brief): a single full-screen fragment shader standing in for the "3D
// scene" reference — an observatory's instrument reading the sky, not a
// rendered object. Deliberately NOT react-three-fiber: this is one
// fullscreen triangle and ~15 lines of GLSL, so its GPU cost is fixed
// regardless of feed length, and it composites *behind* every opaque card
// (z-index 0 — see AmbientField.module.css and layout.tsx's `.app`).
//
// Degrades in three independent ways, each cheaper than the last:
//   1. prefers-reduced-motion -> WebGL is never even initialized; the CSS
//      gradient fallback (always rendered, see .fallback) is the whole UI.
//   2. No WebGL2 context (old device, some privacy modes) -> same CSS
//      fallback, canvas never becomes visible.
//   3. WebGL runs, but the tab is hidden or the battery is low -> the
//      render loop is paused (rAF cancelled) without tearing anything down,
//      resuming when the tab is visible / battery recovers.
//
// Colours are read from the CSS custom properties (--color-ambient-*) via
// getComputedStyle, never hardcoded here — keeps this file clean of the
// no-hardcoded-hex lint rule and makes the shader repaint correctly on
// every theme change for free.
import { useEffect, useRef, useState } from "react";
import styles from "./AmbientField.module.css";

const VERTEX_SRC = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const FRAGMENT_SRC = `
precision mediump float;
varying vec2 vUv;
uniform float uTime;
uniform vec2 uResolution;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
  float v = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 3; i++) {
    v += amp * valueNoise(p);
    p *= 2.02;
    amp *= 0.5;
  }
  return v;
}

void main() {
  vec2 uv = vUv;
  uv.x *= uResolution.x / uResolution.y;
  float t = uTime * 0.015;
  float n1 = fbm(uv * 1.3 + vec2(t, -t * 0.6));
  float n2 = fbm(uv * 1.7 - vec2(t * 0.5, t * 0.35) + 4.2);
  vec3 col = mix(uColorA, uColorB, smoothstep(0.2, 0.85, n1));
  col = mix(col, uColorC, smoothstep(0.35, 0.9, n2) * 0.55);
  gl_FragColor = vec4(col, 1.0);
}
`;

// Render target is deliberately low-res (fixed, story-count-independent
// cost) and let the CSS `width:100%/height:100%` on the canvas upscale it
// — the drift is soft by design, so the blur costs nothing visually.
const RESOLUTION_SCALE = 0.35;
const TARGET_FPS = 30;
const FRAME_BUDGET_MS = 1000 / TARGET_FPS;
const LOW_BATTERY_THRESHOLD = 0.15;

function hexToVec3(hex: string): [number, number, number] {
  const clean = hex.trim().replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  return [r || 0, g || 0, b || 0];
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export function AmbientField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) return; // CSS fallback only — see module docstring.

    const gl = (canvas.getContext("webgl2") ||
      canvas.getContext("webgl")) as WebGLRenderingContext | null;
    if (!gl) return; // CSS fallback only.

    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]), // oversized triangle covers the viewport
      gl.STATIC_DRAW,
    );
    const aPos = gl.getAttribLocation(program, "aPos");

    const uTime = gl.getUniformLocation(program, "uTime");
    const uResolution = gl.getUniformLocation(program, "uResolution");
    const uColorA = gl.getUniformLocation(program, "uColorA");
    const uColorB = gl.getUniformLocation(program, "uColorB");
    const uColorC = gl.getUniformLocation(program, "uColorC");

    // Read fresh on every rendered frame (throttled to 30fps below, so
    // this is ~30 getComputedStyle reads/sec at most) rather than once at
    // setup. That sidesteps a real ordering hazard: this effect and
    // ThemeProvider's own effect (the one that writes `data-theme` to
    // <html>, in lib/theme.tsx) both fire in the same passive-effect flush
    // when the resolved theme changes, and React runs child effects before
    // parent effects — so a one-time read here could observe the
    // *previous* theme's tokens for a frame. Reading per-frame instead
    // means the very next animation frame (after the commit has painted)
    // always reflects the current attribute, with no coordination needed.
    function readThemeColors() {
      // No literal fallback here on purpose (see the no-hardcoded-hex
      // lint rule) — --color-ambient-* is always defined on :root in
      // globals.css, light and dark alike, so getPropertyValue never
      // actually returns "".
      const style = getComputedStyle(document.documentElement);
      return {
        a: hexToVec3(style.getPropertyValue("--color-ambient-1")),
        b: hexToVec3(style.getPropertyValue("--color-ambient-2")),
        c: hexToVec3(style.getPropertyValue("--color-ambient-3")),
      };
    }

    let width = 0;
    let height = 0;
    function resize() {
      const c = canvasRef.current;
      if (!c) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, Math.floor(c.clientWidth * dpr * RESOLUTION_SCALE));
      height = Math.max(1, Math.floor(c.clientHeight * dpr * RESOLUTION_SCALE));
      c.width = width;
      c.height = height;
      gl!.viewport(0, 0, width, height);
    }
    resize();

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => resize());
      ro.observe(canvas);
    } else {
      window.addEventListener("resize", resize);
    }

    let raf = 0;
    let running = true;
    let lastFrame = 0;
    let elapsed = 0;
    const start = performance.now();

    // Measured cost (Playwright + CDP Performance/Emulation domains,
    // simulated mid-tier device via 4x CPU throttling, RTX 4050 laptop
    // GPU host): scrolling the feed with this loop running costs ~6.8
    // percentage points of additional main-thread busy time versus the
    // CSS-only fallback (23.0% vs 16.3% over a 3s scroll window), and
    // raises dropped frames (>50ms) from 3/80 to 7/80 under that
    // aggressive throttle. Idle frame time is flat at ~16.5ms average
    // thanks to the 30fps cap below — see ui-pwa-report.md for the full
    // methodology and the decision to keep it at this cost.
    function frame(now: number) {
      if (!running) return;
      raf = requestAnimationFrame(frame);
      const delta = now - lastFrame;
      if (delta < FRAME_BUDGET_MS) return;
      lastFrame = now;
      elapsed = (now - start) / 1000;
      const colors = readThemeColors();

      gl!.useProgram(program);
      gl!.bindBuffer(gl!.ARRAY_BUFFER, posBuffer);
      gl!.enableVertexAttribArray(aPos);
      gl!.vertexAttribPointer(aPos, 2, gl!.FLOAT, false, 0, 0);
      gl!.uniform1f(uTime, elapsed);
      gl!.uniform2f(uResolution, width, height);
      gl!.uniform3f(uColorA, ...colors.a);
      gl!.uniform3f(uColorB, ...colors.b);
      gl!.uniform3f(uColorC, ...colors.c);
      gl!.drawArrays(gl!.TRIANGLES, 0, 3);
    }

    function pause() {
      running = false;
      cancelAnimationFrame(raf);
    }
    function resume() {
      if (running) return;
      running = true;
      lastFrame = 0;
      raf = requestAnimationFrame(frame);
    }

    function onVisibility() {
      if (document.hidden) pause();
      else resume();
    }
    document.addEventListener("visibilitychange", onVisibility);

    // Battery Status API is Chromium-only and behind no flag; absence
    // (Safari, Firefox) just means this guard never fires — the loop
    // still respects tab visibility and reduced-motion regardless.
    let batteryCleanup: (() => void) | null = null;
    // getBattery() is async and can resolve *after* this effect has
    // already been cleaned up (React StrictMode's dev-only double-invoke
    // makes this deterministic, but it's a real race in production too —
    // any sufficiently fast unmount could hit it). Without this guard, a
    // late resolution would register fresh listeners and call resume()
    // on a closure whose GL resources cleanup already deleted, reviving a
    // dead render loop that then errors on every frame. teardown below
    // flips this before anything else runs.
    let unmounted = false;
    const nav = navigator as Navigator & {
      getBattery?: () => Promise<{
        level: number;
        charging: boolean;
        addEventListener: (type: string, cb: () => void) => void;
        removeEventListener: (type: string, cb: () => void) => void;
      }>;
    };
    if (nav.getBattery) {
      nav.getBattery().then((battery) => {
        if (unmounted) return;
        function evaluate() {
          if (battery.level < LOW_BATTERY_THRESHOLD && !battery.charging) pause();
          else if (!document.hidden) resume();
        }
        battery.addEventListener("levelchange", evaluate);
        battery.addEventListener("chargingchange", evaluate);
        evaluate();
        batteryCleanup = () => {
          battery.removeEventListener("levelchange", evaluate);
          battery.removeEventListener("chargingchange", evaluate);
        };
      });
    }

    raf = requestAnimationFrame(frame);
    setReady(true);

    return () => {
      unmounted = true;
      running = false;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", resize);
      batteryCleanup?.();
      gl!.deleteProgram(program);
      gl!.deleteShader(vertexShader);
      gl!.deleteShader(fragmentShader);
      gl!.deleteBuffer(posBuffer);
    };
    // Mount once. Theme changes are picked up per-frame (readThemeColors
    // above), not by re-running setup — see that function's docstring.
  }, []);

  return (
    <div className={styles.root} aria-hidden="true">
      <div className={styles.fallback} />
      <canvas ref={canvasRef} className={`${styles.canvas} ${ready ? styles.canvasReady : ""}`} />
    </div>
  );
}
