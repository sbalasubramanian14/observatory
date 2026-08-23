// Feed view mode: "list" (dense, scannable — the default, and the only
// mode on desktop) vs "cards" (mobile full-viewport one-story-at-a-time
// deck, StoryDeck.tsx). Persisted per-device, same pattern as sort.ts.
// Default is "list" even on mobile: it's the only mode that hydrates
// identically to the server-rendered snapshot (no viewport-based guess
// that could mismatch), and the toggle to switch is one tap, right next
// to the sort control.
"use client";

import { useCallback, useSyncExternalStore } from "react";

export type ViewMode = "list" | "cards";

const STORAGE_KEY = "observatory:viewMode";
const LOCAL_SYNC_EVENT = "observatory:viewMode-sync";

function isValidMode(v: string | null): v is ViewMode {
  return v === "list" || v === "cards";
}

export function getViewMode(): ViewMode {
  if (typeof window === "undefined") return "list";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return isValidMode(stored) ? stored : "list";
}

export function setViewMode(mode: ViewMode) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, mode);
  window.dispatchEvent(new Event(LOCAL_SYNC_EVENT));
}

function subscribe(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(LOCAL_SYNC_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(LOCAL_SYNC_EVENT, callback);
  };
}

function getServerSnapshot(): ViewMode {
  return "list";
}

export function useViewMode(): [ViewMode, (mode: ViewMode) => void] {
  const mode = useSyncExternalStore(subscribe, getViewMode, getServerSnapshot);
  const set = useCallback((m: ViewMode) => setViewMode(m), []);
  return [mode, set];
}
