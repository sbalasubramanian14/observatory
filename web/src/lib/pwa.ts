// PWA runtime: service worker registration/update handling, online status,
// and "last synced" tracking for the offline banner's honest timestamp.
// Same useSyncExternalStore pattern as theme.tsx/sort.ts/sourceFilter.ts —
// a same-tab sync event, since localStorage's own `storage` event never
// fires in the tab that made the write.
"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

function readOnline(): boolean {
  if (typeof navigator === "undefined") return true;
  return navigator.onLine;
}

function subscribeOnline(callback: () => void) {
  window.addEventListener("online", callback);
  window.addEventListener("offline", callback);
  return () => {
    window.removeEventListener("online", callback);
    window.removeEventListener("offline", callback);
  };
}

function getServerOnlineSnapshot(): boolean {
  return true;
}

// A second, independent signal for "should the offline banner show",
// alongside navigator.onLine: whether the most recent manifest fetch
// actually had to fall back to the service worker's cache (see sw.js's
// networkFirst, and the X-Observatory-From-Cache header it sets). This
// matters because navigator.onLine is a coarse link-layer flag that some
// browsers/automation environments update unreliably around a navigation
// (it can still read `true` immediately after a hard reload performed
// while genuinely offline) -- direct evidence that a real fetch just
// failed and had to be served from cache is a strictly stronger signal
// than trusting the browser's own flag alone. In-memory only (not
// persisted): a fresh page load always re-evaluates from its own first
// getManifest() call rather than remembering a stale verdict.
let servingFromCache = false;
const FROM_CACHE_EVENT = "observatory:fromCache-sync";

export function recordServedFrom(fromCache: boolean) {
  if (servingFromCache === fromCache) return;
  servingFromCache = fromCache;
  if (typeof window !== "undefined") window.dispatchEvent(new Event(FROM_CACHE_EVENT));
}

function readServingFromCache(): boolean {
  return servingFromCache;
}

function subscribeServingFromCache(callback: () => void) {
  window.addEventListener(FROM_CACHE_EVENT, callback);
  return () => window.removeEventListener(FROM_CACHE_EVENT, callback);
}

function getServerServingFromCacheSnapshot(): boolean {
  return false;
}

export function useServingFromCache(): boolean {
  return useSyncExternalStore(
    subscribeServingFromCache,
    readServingFromCache,
    getServerServingFromCacheSnapshot,
  );
}

const LAST_SYNC_KEY = "observatory:lastSync";
const LAST_SYNC_EVENT = "observatory:lastSync-sync";

/** Called by lib/bundle.ts's getManifest() on every successful fetch of the
 * *live* manifest — the one signal that means "we genuinely reached the
 * network just now", as opposed to a service-worker cache hit. */
export function recordSync() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LAST_SYNC_KEY, String(Date.now()));
    window.dispatchEvent(new Event(LAST_SYNC_EVENT));
  } catch {
    // best-effort; the offline banner just won't have a timestamp
  }
}

function readLastSync(): number | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(LAST_SYNC_KEY);
  return raw ? Number(raw) : null;
}

function subscribeLastSync(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(LAST_SYNC_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(LAST_SYNC_EVENT, callback);
  };
}

export function useLastSync(): number | null {
  return useSyncExternalStore(subscribeLastSync, readLastSync, () => null);
}

/** True once the browser has fired 'offline' and not yet 'online' again.
 * navigator.onLine is used for the initial value; it's a coarse signal
 * (link-layer connectivity, not "can reach our origin") but is exactly
 * right for "is this a good moment to show the honest offline banner",
 * and is corrected instantly by the events either way. */
export function useOnlineStatus(): boolean {
  // useSyncExternalStore (not useState+useEffect) is the correct tool here
  // — same reasoning as theme.tsx's ThemeProvider: it reads navigator.onLine
  // directly, gives a safe server snapshot (true — never show "offline" on
  // the prerendered HTML) for the static export build, and lets React
  // reconcile a client value that differs from the server snapshot without
  // a hydration-mismatch warning, which a plain useState lazy initializer
  // reading `navigator` cannot do safely.
  return useSyncExternalStore(subscribeOnline, readOnline, getServerOnlineSnapshot);
}

interface SwUpdateState {
  updateAvailable: boolean;
  applyUpdate: () => void;
}

/** Registers /sw.js once, then watches for a new worker reaching the
 * "installed" (i.e. waiting) state, which means a new bundle of the app
 * shell is ready but the current tab is still controlled by the old one.
 * `applyUpdate` tells the waiting worker to skip waiting and reloads once
 * it takes control — the reader chooses the moment (UpdateToast.tsx),
 * this never happens silently underneath a mid-read. */
export function useServiceWorker(): SwUpdateState {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);

  // Tracks whether *this* client explicitly asked a waiting worker to
  // skip waiting (see applyUpdate below) — the only case that should
  // reload the page. 'controllerchange' also fires the very first time a
  // service worker ever activates for this page (going from no controller
  // to controlled), which is not an update and must never reload; without
  // this guard every first-time visitor would get an unexpected reload
  // moments after landing, sometimes mid-navigation.
  const awaitingReload = useRef(false);

  useEffect(() => {
    if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;

    let cancelled = false;

    function onControllerChange() {
      if (!awaitingReload.current) return;
      window.location.reload();
    }

    navigator.serviceWorker.register("/sw.js").then((registration) => {
      if (cancelled) return;

      if (registration.waiting && navigator.serviceWorker.controller) {
        setWaitingWorker(registration.waiting);
        setUpdateAvailable(true);
      }

      registration.addEventListener("updatefound", () => {
        const installing = registration.installing;
        if (!installing) return;
        installing.addEventListener("statechange", () => {
          if (installing.state === "installed" && navigator.serviceWorker.controller) {
            setWaitingWorker(installing);
            setUpdateAvailable(true);
          }
        });
      });
    });

    navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);

    return () => {
      cancelled = true;
      navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
    };
  }, []);

  const applyUpdate = useCallback(() => {
    awaitingReload.current = true;
    waitingWorker?.postMessage("SKIP_WAITING");
  }, [waitingWorker]);

  return { updateAvailable, applyUpdate };
}
