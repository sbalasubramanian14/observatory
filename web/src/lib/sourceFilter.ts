// Source/territory filter (multi-select). Persisted per-device in
// localStorage, same pattern as sort.ts and personalization.tsx: a
// same-tab sync event, since localStorage's own `storage` event never
// fires in the tab that made the write.
"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { FeedStoryRow } from "./types";

const STORAGE_KEY = "observatory:sourceFilter:sourceIds";
const LOCAL_SYNC_EVENT = "observatory:sourceFilter-sync";

function isBrowser() {
  return typeof window !== "undefined";
}

// useSyncExternalStore requires getSnapshot to return a referentially
// stable value when nothing has changed (otherwise React logs "the
// result of getSnapshot should be cached" and can loop). A plain
// `new Set(JSON.parse(...))` on every call would violate that, so the
// parsed Set is cached and only rebuilt when the underlying raw string
// actually differs from what produced the cached value.
let cachedRaw: string | null | undefined = undefined;
let cachedSet: Set<string> = new Set();
const EMPTY_SET: Set<string> = new Set();

function readSelected(): Set<string> {
  if (!isBrowser()) return EMPTY_SET;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw === cachedRaw) return cachedSet;
  cachedRaw = raw;
  try {
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    cachedSet = new Set(
      Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [],
    );
  } catch {
    cachedSet = new Set();
  }
  return cachedSet;
}

function writeSelected(ids: Set<string>) {
  if (!isBrowser()) return;
  const arr = Array.from(ids).sort();
  const raw = JSON.stringify(arr);
  try {
    window.localStorage.setItem(STORAGE_KEY, raw);
  } catch {
    // localStorage can throw (quota, private mode) -- the filter is a
    // best-effort convenience, never load-bearing, so we swallow this
    // (same rationale as personalization.tsx's writeJson).
  }
  cachedRaw = raw;
  cachedSet = new Set(arr);
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

function getServerSnapshot(): Set<string> {
  return EMPTY_SET;
}

/** The persisted set of selected source ids. Empty means "no filter" --
 * see storyMatchesFilter. Territory checkboxes in the UI are just a
 * convenience that expand/collapse to every source id in that territory;
 * there is only ever one underlying selection (source ids), which keeps
 * "filter by source" and "filter by territory" from becoming two
 * independent axes a reader would have to reason about combining. */
export function useSourceFilter(): [Set<string>, (ids: Set<string>) => void] {
  const selected = useSyncExternalStore(subscribe, readSelected, getServerSnapshot);
  const setSelected = useCallback((ids: Set<string>) => writeSelected(ids), []);
  return [selected, setSelected];
}

export function toggleId(current: Set<string>, id: string): Set<string> {
  const next = new Set(current);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

/** Adds (select=true) or removes (select=false) every id in `ids` --
 * used for the territory checkbox's "select/deselect all its sources". */
export function setMany(current: Set<string>, ids: string[], select: boolean): Set<string> {
  const next = new Set(current);
  for (const id of ids) {
    if (select) next.add(id);
    else next.delete(id);
  }
  return next;
}

/**
 * A story can contain items from multiple sources (spec: a cluster
 * merges coverage of the same event across outlets). This matches on
 * ANY of a story's contributing sources being selected, not ALL of them:
 * a story is relevant to a selected territory/source as soon as ONE of
 * its items came from there, and requiring every item to match would
 * hide genuinely on-topic multi-source stories just because a wire-copy
 * mirror or an unrelated-territory outlet also picked it up -- the
 * opposite of what a reader narrowing to "policy" wants. Empty selection
 * means no filter is active, so everything matches (never accidentally
 * hides the whole feed).
 */
export function storyMatchesFilter(story: FeedStoryRow, selected: Set<string>): boolean {
  if (selected.size === 0) return true;
  return story.source_ids.some((id) => selected.has(id));
}
