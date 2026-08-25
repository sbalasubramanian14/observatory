// Client-side personalization (spec §4.3). Everything here lives in
// localStorage and never leaves the device: positive/negative centroids in
// embedding space, read/saved/dismissed story ids, and the personalization
// toggle. Nothing here makes a network request.
"use client";

import { createContext, useCallback, useContext, useMemo, useSyncExternalStore } from "react";
import type { FeedStoryRow } from "./types";

const NS = "observatory:";
const KEYS = {
  enabled: `${NS}personalization:enabled`,
  positiveSum: `${NS}centroid:positiveSum`,
  positiveCount: `${NS}centroid:positiveCount`,
  negativeSum: `${NS}centroid:negativeSum`,
  negativeCount: `${NS}centroid:negativeCount`,
  embeddingModelId: `${NS}embeddingModelId`,
  read: `${NS}read`,
  saved: `${NS}saved`,
  savedSnapshots: `${NS}saved:snapshots`,
  dismissed: `${NS}dismissed`,
} as const;

// Signals feed the positive or negative centroid with different weight:
// a save is a stronger declaration of interest than a passive open, and a
// dismissal is a full negative signal.
const SIGNAL_WEIGHT: Record<Signal, number> = {
  open: 1,
  save: 2.5,
  dismiss: 1,
};

export type Signal = "open" | "save" | "dismiss";

function isBrowser() {
  return typeof window !== "undefined";
}

function readJson<T>(key: string, fallback: T): T {
  if (!isBrowser()) return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw == null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown) {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // localStorage can throw (quota, private mode) — personalization is a
    // best-effort enhancement, never load-bearing, so we swallow this.
  }
}

// ---- toggle -----------------------------------------------------------

// Same rationale as src/lib/theme.tsx: localStorage's `storage` event never
// fires in the tab that made the write, so the toggle in the header and the
// re-rank logic on the feed page need their own same-tab signal to agree.
const LOCAL_SYNC_EVENT = "observatory:personalization-sync";

/** Default is OFF on a fresh client (spec §4.3): the reader sees the
 * published importance order before any re-ranking is applied to it. */
export function isPersonalizationEnabled(): boolean {
  return readJson<boolean>(KEYS.enabled, false);
}

export function setPersonalizationEnabled(value: boolean) {
  writeJson(KEYS.enabled, value);
  if (isBrowser()) window.dispatchEvent(new Event(LOCAL_SYNC_EVENT));
}

function subscribeEnabled(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(LOCAL_SYNC_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(LOCAL_SYNC_EVENT, callback);
  };
}

function getServerEnabledSnapshot(): boolean {
  return false;
}

interface PersonalizationContextValue {
  enabled: boolean;
  setEnabled: (v: boolean) => void;
}

const PersonalizationContext = createContext<PersonalizationContextValue | null>(null);

/** Wraps the app so the toggle in the header and the re-rank logic on the
 * feed page always read the same value. Backed by useSyncExternalStore
 * (not useState+useEffect) so the static export's prerendered HTML gets a
 * safe default and the browser reconciles from localStorage without ever
 * calling setState synchronously inside an effect. */
export function PersonalizationProvider({ children }: { children: React.ReactNode }) {
  const enabled = useSyncExternalStore(
    subscribeEnabled,
    isPersonalizationEnabled,
    getServerEnabledSnapshot,
  );

  const setEnabled = useCallback((v: boolean) => {
    setPersonalizationEnabled(v);
  }, []);

  const value = useMemo(() => ({ enabled, setEnabled }), [enabled, setEnabled]);

  return (
    <PersonalizationContext.Provider value={value}>{children}</PersonalizationContext.Provider>
  );
}

export function usePersonalization(): PersonalizationContextValue {
  const ctx = useContext(PersonalizationContext);
  if (!ctx) throw new Error("usePersonalization must be used within a PersonalizationProvider");
  return ctx;
}

// ---- read / saved / dismissed sets ------------------------------------

// Same same-tab-sync gap as the toggle above: writing to localStorage
// never fires `storage` in the tab that wrote it, so the saved-count badge
// in the header and the /saved and /dismissed pages need their own signal
// to notice a markSaved/unmarkSaved/markDismissed/unmarkDismissed call
// made elsewhere on the same page (e.g. a StoryCard's Save button).
const ID_SET_SYNC_EVENT = "observatory:idset-sync";

function readIdSet(key: string): Set<number> {
  return new Set(readJson<number[]>(key, []));
}

/** Raw stored order (oldest first — each add appends), NOT deduplicated
 * into a Set's insertion order guarantee by accident: JS Sets do preserve
 * insertion order, but reading straight from the JSON array is the more
 * direct way to expose "the order things were saved in" to callers like
 * the /saved page, which reverses it for "most recently saved first". */
function readIdListOrdered(key: string): number[] {
  return readJson<number[]>(key, []);
}

function addToIdSet(key: string, id: number) {
  const set = readIdSet(key);
  set.add(id);
  writeJson(key, Array.from(set));
  if (isBrowser()) window.dispatchEvent(new Event(ID_SET_SYNC_EVENT));
}

function removeFromIdSet(key: string, id: number) {
  const set = readIdSet(key);
  set.delete(id);
  writeJson(key, Array.from(set));
  if (isBrowser()) window.dispatchEvent(new Event(ID_SET_SYNC_EVENT));
}

export function getReadIds(): Set<number> {
  return readIdSet(KEYS.read);
}

export function markRead(id: number) {
  addToIdSet(KEYS.read, id);
}

export function getSavedIds(): Set<number> {
  return readIdSet(KEYS.saved);
}

export function getSavedIdsOrdered(): number[] {
  return readIdListOrdered(KEYS.saved);
}

export function getDismissedIds(): Set<number> {
  return readIdSet(KEYS.dismissed);
}

export function getDismissedIdsOrdered(): number[] {
  return readIdListOrdered(KEYS.dismissed);
}

// ---- saved-story snapshots --------------------------------------------
//
// The bundle only carries a rolling window of recent news ([publish]
// .retention_days, currently 5 days). A saved story that ages out of that
// window disappears from the feed pages, and /saved — which resolves ids
// against those pages — would render it as a "no longer in the bundle"
// placeholder. That makes the bookmark button a promise the app cannot
// keep, at any window size; narrowing the window to 5 days just makes it
// happen within the week rather than within the quarter.
//
// So saving keeps its own copy of the card. It is the same FeedStoryRow
// the bundle published — titles, links and Observatory's own generated
// summary, never article text (spec §4.2) — and it stays on the device
// exactly like every other signal here.

type SavedSnapshots = Record<string, FeedStoryRow>;

function readSnapshots(): SavedSnapshots {
  return readJson<SavedSnapshots>(KEYS.savedSnapshots, {});
}

/** The saved card as it was when it was saved, for stories the current
 * bundle no longer carries. Returns null when there is no snapshot —
 * bookmarks made before this existed have none, and must still degrade to
 * the placeholder row rather than crash. */
export function getSavedSnapshot(id: number): FeedStoryRow | null {
  return readSnapshots()[String(id)] ?? null;
}

/** Re-capture snapshots for saved stories the bundle still carries, so a
 * story saved before enrichment ran (title only, "No summary yet") is not
 * frozen that way forever. One write per /saved visit, and only when
 * something actually changed. */
export function refreshSavedSnapshots(rows: readonly FeedStoryRow[]) {
  const saved = getSavedIds();
  if (saved.size === 0) return;
  const snapshots = readSnapshots();
  let changed = false;
  for (const row of rows) {
    if (!saved.has(row.id)) continue;
    const key = String(row.id);
    if (JSON.stringify(snapshots[key]) === JSON.stringify(row)) continue;
    snapshots[key] = row;
    changed = true;
  }
  if (changed) writeJson(KEYS.savedSnapshots, snapshots);
}

export function markSaved(id: number, snapshot?: FeedStoryRow) {
  addToIdSet(KEYS.saved, id);
  if (snapshot) {
    const snapshots = readSnapshots();
    snapshots[String(id)] = snapshot;
    writeJson(KEYS.savedSnapshots, snapshots);
  }
}

export function unmarkSaved(id: number) {
  removeFromIdSet(KEYS.saved, id);
  // Unsaving drops the copy too: keeping it would grow localStorage
  // without bound for a story the reader has explicitly let go of.
  const snapshots = readSnapshots();
  if (String(id) in snapshots) {
    delete snapshots[String(id)];
    writeJson(KEYS.savedSnapshots, snapshots);
  }
}

export function markDismissed(id: number) {
  addToIdSet(KEYS.dismissed, id);
}

/** D3: undo an accidental dismiss. Only removes the id from the dismissed
 * set — the story reappears in the main feed on its own next time (if it
 * is still in the current bundle window and not also marked read). */
export function unmarkDismissed(id: number) {
  removeFromIdSet(KEYS.dismissed, id);
}

function subscribeIdSets(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(ID_SET_SYNC_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(ID_SET_SYNC_EVENT, callback);
  };
}

function getServerCountSnapshot(): number {
  return 0;
}

/** Live count of saved stories, for the header badge (D2). */
export function useSavedCount(): number {
  return useSyncExternalStore(
    subscribeIdSets,
    () => getSavedIds().size,
    getServerCountSnapshot,
  );
}

/** Live count of dismissed stories, for the /dismissed page link (D3). */
export function useDismissedCount(): number {
  return useSyncExternalStore(
    subscribeIdSets,
    () => getDismissedIds().size,
    getServerCountSnapshot,
  );
}

// ---- centroids ----------------------------------------------------------

function zeros(dims: number): number[] {
  return new Array(dims).fill(0);
}

/** The bundle's embedding_model_id is a client-visible contract (spec
 * §3.3): vectors from different models aren't comparable, so if the model
 * changes underneath a returning reader we must discard their centroids
 * rather than silently corrupt fit scores. */
function ensureModelConsistency(modelId: string) {
  const stored = isBrowser() ? window.localStorage.getItem(KEYS.embeddingModelId) : null;
  if (stored !== null && stored !== modelId) {
    resetCentroids();
  }
  if (isBrowser()) {
    window.localStorage.setItem(KEYS.embeddingModelId, modelId);
  }
}

export function resetCentroids() {
  if (!isBrowser()) return;
  [KEYS.positiveSum, KEYS.positiveCount, KEYS.negativeSum, KEYS.negativeCount].forEach(
    (k) => window.localStorage.removeItem(k),
  );
}

export function recordSignal(
  signal: Signal,
  storyEmbedding: Float32Array,
  embeddingModelId: string,
) {
  if (!isBrowser()) return;
  ensureModelConsistency(embeddingModelId);

  const dims = storyEmbedding.length;
  const isPositive = signal === "open" || signal === "save";
  const sumKey = isPositive ? KEYS.positiveSum : KEYS.negativeSum;
  const countKey = isPositive ? KEYS.positiveCount : KEYS.negativeCount;
  const weight = SIGNAL_WEIGHT[signal];

  const sum = readJson<number[]>(sumKey, zeros(dims));
  for (let i = 0; i < dims; i++) {
    sum[i] += storyEmbedding[i] * weight;
  }
  const count = readJson<number>(countKey, 0) + weight;

  writeJson(sumKey, sum);
  writeJson(countKey, count);
}

function normalize(vec: Float32Array): Float32Array {
  let mag = 0;
  for (let i = 0; i < vec.length; i++) mag += vec[i] * vec[i];
  mag = Math.sqrt(mag);
  if (mag === 0) return vec;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / mag;
  return out;
}

export interface Centroids {
  positive: Float32Array | null;
  negative: Float32Array | null;
}

export function getCentroids(): Centroids {
  const posCount = readJson<number>(KEYS.positiveCount, 0);
  const negCount = readJson<number>(KEYS.negativeCount, 0);
  const positive =
    posCount > 0
      ? normalize(Float32Array.from(readJson<number[]>(KEYS.positiveSum, [])))
      : null;
  const negative =
    negCount > 0
      ? normalize(Float32Array.from(readJson<number[]>(KEYS.negativeSum, [])))
      : null;
  return { positive, negative };
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/** fit = cos(story, positive) − cos(story, negative), per spec §4.3 step 2.
 * A story with no centroid data yet, or no embedding available, is neutral
 * (fit 0) rather than penalized. */
export function computeFit(
  storyEmbedding: Float32Array | undefined,
  centroids: Centroids,
): number {
  if (!storyEmbedding) return 0;
  const pos = centroids.positive ? cosineSimilarity(storyEmbedding, centroids.positive) : 0;
  const neg = centroids.negative ? cosineSimilarity(storyEmbedding, centroids.negative) : 0;
  return pos - neg;
}

// ---- re-ranking ---------------------------------------------------------

export interface RankedStory {
  story: FeedStoryRow;
  fit: number;
  combined: number;
  /** True for a story placed in one of the reserved serendipity slots:
   * high published importance, low personal fit, kept visible on purpose
   * (spec §4.3: "reserves 15% of feed slots... visibly labelled"). */
  reserved: boolean;
}

const RESERVED_FRACTION = 0.15;
// Weight given to personal fit vs. published importance when blending.
// Keeping importance at >= 0.5 means personalization nudges order rather
// than replacing the editorial signal outright.
const FIT_WEIGHT = 0.5;
const IMPORTANCE_WEIGHT = 1 - FIT_WEIGHT;

function minMax(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

function normalizeTo01(value: number, min: number, max: number): number {
  return (value - min) / (max - min);
}

/**
 * Re-ranks an importance-ordered story list.
 *
 * `enabled: false` renders the published order untouched (only already-read
 * filtering applies) — spec §4.3 "Personalization is switchable, per
 * client": off is not a separate code path, it's the absence of the re-rank
 * step below.
 */
export function rerankFeed(
  stories: FeedStoryRow[],
  embeddings: Map<number, Float32Array>,
  centroids: Centroids,
  enabled: boolean,
): RankedStory[] {
  const readIds = getReadIds();
  const dismissedIds = getDismissedIds();
  const visible = stories.filter((s) => !readIds.has(s.id) && !dismissedIds.has(s.id));

  if (!enabled) {
    return visible.map((story) => ({ story, fit: 0, combined: story.score, reserved: false }));
  }

  const fits = visible.map((s) => computeFit(embeddings.get(s.id), centroids));
  const [fitMin, fitMax] = minMax(fits);
  const importances = visible.map((s) => s.score);
  const [impMin, impMax] = minMax(importances);

  const scored = visible.map((story, i) => {
    const fit = fits[i];
    const normFit = normalizeTo01(fit, fitMin, fitMax);
    const normImportance = normalizeTo01(story.score, impMin, impMax);
    const combined = IMPORTANCE_WEIGHT * normImportance + FIT_WEIGHT * normFit;
    return { story, fit, normFit, normImportance, combined };
  });

  const total = scored.length;
  const reserveCount = Math.round(total * RESERVED_FRACTION);

  // Reserved pool: high importance, low fit, ranked by importance so the
  // most important "outside your usual" stories win the reserved slots.
  const byImportanceDesc = [...scored].sort((a, b) => b.normImportance - a.normImportance);
  const lowFitThreshold = 0.4; // bottom ~40% of normalized fit counts as "low fit"
  const reservedCandidates = byImportanceDesc.filter((s) => s.normFit <= lowFitThreshold);
  const reservedPicks = new Set(
    (reservedCandidates.length >= reserveCount ? reservedCandidates : byImportanceDesc).slice(
      0,
      reserveCount,
    ),
  );

  const mainOrder = [...scored]
    .filter((s) => !reservedPicks.has(s))
    .sort((a, b) => b.combined - a.combined);

  const reservedOrder = Array.from(reservedPicks).sort(
    (a, b) => b.normImportance - a.normImportance,
  );

  // Interleave reserved picks evenly through the main order so they read as
  // part of the feed rather than dumped at the end.
  const result: RankedStory[] = [];
  const step =
    reservedOrder.length > 0
      ? Math.max(1, Math.floor(mainOrder.length / (reservedOrder.length + 1)))
      : Infinity;
  let mainIdx = 0;
  let reservedIdx = 0;
  let sincePlacement = 0;
  while (mainIdx < mainOrder.length || reservedIdx < reservedOrder.length) {
    if (mainIdx < mainOrder.length) {
      const s = mainOrder[mainIdx++];
      result.push({ story: s.story, fit: s.fit, combined: s.combined, reserved: false });
      sincePlacement++;
    }
    if (reservedIdx < reservedOrder.length && sincePlacement >= step) {
      const s = reservedOrder[reservedIdx++];
      result.push({ story: s.story, fit: s.fit, combined: s.combined, reserved: true });
      sincePlacement = 0;
    }
  }
  // Any leftover reserved picks (short main list) go at the end.
  while (reservedIdx < reservedOrder.length) {
    const s = reservedOrder[reservedIdx++];
    result.push({ story: s.story, fit: s.fit, combined: s.combined, reserved: true });
  }

  return result;
}
