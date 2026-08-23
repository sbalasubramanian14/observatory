// D1: the sort control. Persisted per-device in localStorage, same pattern
// as theme.tsx and personalization.tsx (a same-tab sync event, since
// localStorage's own `storage` event never fires in the writing tab).
"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { FeedStoryRow } from "./types";

export type SortMode = "recency" | "importance" | "outlets" | "unread";

export const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: "recency", label: "Recency" },
  { value: "importance", label: "Importance" },
  { value: "outlets", label: "Most outlets" },
  { value: "unread", label: "Unread first" },
];

const STORAGE_KEY = "observatory:sort";
const LOCAL_SYNC_EVENT = "observatory:sort-sync";

function isValidMode(v: string | null): v is SortMode {
  return v === "recency" || v === "importance" || v === "outlets" || v === "unread";
}

/** Default is "recency" (D1: changed from importance per the owner's
 * explicit request) on a fresh client. */
export function getSortMode(): SortMode {
  if (typeof window === "undefined") return "recency";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return isValidMode(stored) ? stored : "recency";
}

export function setSortMode(mode: SortMode) {
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

function getServerSnapshot(): SortMode {
  return "recency";
}

export function useSortMode(): [SortMode, (mode: SortMode) => void] {
  const mode = useSyncExternalStore(subscribe, getSortMode, getServerSnapshot);
  const set = useCallback((m: SortMode) => setSortMode(m), []);
  return [mode, set];
}

function byRecencyDesc(a: FeedStoryRow, b: FeedStoryRow): number {
  return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
}

/**
 * Literal, non-personalized ordering for the three sort modes that are NOT
 * "Importance". See docs/superpowers/sdd/phaseD-report.md for the full
 * rationale; short version: a reader who explicitly picks "Recency" (or
 * "Most outlets", or "Unread first") is asking to see that literal order,
 * so these modes deliberately do NOT apply the fit-based personalization
 * re-rank or the serendipity reservation — those are specific to
 * "Importance" (spec §4.3 frames personalization as re-ranking "the
 * importance-ordered feed"). A sort control that silently got overridden
 * by fit-blending would contradict what it says it does.
 *
 * The always-on read/dismissed filtering (spec §4.3 step 4) still applies
 * here, with one deliberate exception: "unread" mode itself shows BOTH
 * read and unread stories (unread ones first, each group sorted by
 * recency) — that's the whole point of a "catch up" mode, and hiding read
 * stories from it would make it identical to "recency". Dismissed stories
 * stay excluded even in "unread" mode; a dismissal is a deliberate removal
 * (D3 gives it its own recoverable review page), not a passive read marker.
 */
export function sortNonImportance(
  stories: FeedStoryRow[],
  mode: Exclude<SortMode, "importance">,
  readIds: Set<number>,
  dismissedIds: Set<number>,
): FeedStoryRow[] {
  const notDismissed = stories.filter((s) => !dismissedIds.has(s.id));

  if (mode === "unread") {
    const unread = notDismissed.filter((s) => !readIds.has(s.id)).sort(byRecencyDesc);
    const read = notDismissed.filter((s) => readIds.has(s.id)).sort(byRecencyDesc);
    return [...unread, ...read];
  }

  const visible = notDismissed.filter((s) => !readIds.has(s.id));
  if (mode === "recency") return [...visible].sort(byRecencyDesc);
  // mode === "outlets"
  return [...visible].sort((a, b) => b.outlet_count - a.outlet_count || byRecencyDesc(a, b));
}
