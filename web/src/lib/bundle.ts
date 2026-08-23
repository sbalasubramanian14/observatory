import type {
  FeedPage,
  FeedStoryRow,
  Manifest,
  SourcesFile,
  StoryDetail,
} from "./types";
import { recordSync, recordServedFrom } from "./pwa";

// Fetches the published bundle at runtime (never bundled at build time —
// spec §4.5: "Deployed once; it fetches the manifest and bundle at
// runtime"). All paths resolve against a single configurable base URL, in
// this order:
//   1. NEXT_PUBLIC_BUNDLE_URL, if set — explicit override, e.g. to point a
//      preview build at a staging data repo, or to force a local run to
//      read from the CDN.
//   2. "/data" while running `next dev` — the local dev server serves the
//      copy `predev`/sync-bundle.mjs mirrors from the repo root's public/,
//      so iteration doesn't require network access.
//   3. Otherwise (production build — this is what ships to Vercel, where
//      the repo-root public/ does not exist) the published bundle's public
//      Git repo, read via raw.githubusercontent.com.
//
// Why raw.githubusercontent.com and not jsDelivr as the default: jsDelivr's
// gh/ endpoint resolves a branch name (@main) to a commit at its edge and
// only re-resolves roughly every 12h, and that resolution isn't something a
// client request can cache-bust — query strings don't defeat it. A feed
// that republishes every 30 minutes would spend most of its life pointing
// at a commit that's hours stale. raw.githubusercontent.com serves the
// current tip on every request, which is what spec §4.1 requires of
// manifest.json ("small, always fetched fresh"). The tradeoff is a
// stricter, IP-scoped rate limit than a real CDN — acceptable here because
// only two files per session are fetched fresh (manifest.json, sources.json);
// every other request is for a content-addressed filename, which this
// module deliberately fetches *without* cache-busting so the browser's own
// HTTP cache can serve repeat requests with zero network traffic. That's
// the whole point of the manifest-plus-content-hash split in §4.1: pay the
// freshness cost once, on the small file, and let everything else be
// cached forever.
const DEFAULT_REMOTE_BASE =
  "https://raw.githubusercontent.com/sbalasubramanian14/observatory-almanac/main";

const explicitBase = process.env.NEXT_PUBLIC_BUNDLE_URL;
const DATA_BASE = (
  explicitBase && explicitBase.length > 0
    ? explicitBase
    : process.env.NODE_ENV === "development"
      ? "/data"
      : DEFAULT_REMOTE_BASE
).replace(/\/$/, "");

function resolveUrl(path: string): string {
  return `${DATA_BASE}/${path.replace(/^\//, "")}`;
}

/** Fetches a *mutable* bundle file — currently only manifest.json and
 * sources.json, which the pipeline overwrites in place on every run rather
 * than writing under a new content-hashed name. A cache-busting query
 * param plus `cache: "no-store"` guarantees this always reaches the origin,
 * never a stale browser/CDN copy. */
async function fetchFresh<T>(path: string, onResponse?: (res: Response) => void): Promise<T> {
  const url = `${resolveUrl(path)}?t=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  onResponse?.(res);
  return (await res.json()) as T;
}

/** Fetches a *content-addressed* bundle file (feed pages, story details).
 * The hash in the filename guarantees the response body never changes for
 * a given URL, so this deliberately does NOT cache-bust — letting the
 * browser (and any cache in front of the origin) reuse a prior response is
 * exactly what spec §4.1's immutable-filename design is for. */
async function fetchImmutable<T>(path: string): Promise<T> {
  const url = resolveUrl(path);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function getManifest(): Promise<Manifest> {
  // recordSync() only fires for a response that genuinely reached the
  // network just now. sw.js's network-first strategy for manifest.json
  // falls back to its cache when offline and tags that fallback with
  // X-Observatory-From-Cache — otherwise every offline reload would
  // silently rewrite the offline banner's "cached data from <time>" to
  // "now", exactly the dishonest UX the spec rules out. See lib/pwa.ts's
  // useLastSync, which the banner reads.
  return fetchFresh<Manifest>("manifest.json", (res) => {
    const fromCache = !!res.headers.get("X-Observatory-From-Cache");
    if (!fromCache) recordSync();
    // See lib/pwa.ts's useServingFromCache — a second, more direct signal
    // for the offline banner than navigator.onLine alone.
    recordServedFrom(fromCache);
  });
}

export async function getFeedPage(path: string): Promise<FeedPage> {
  return fetchImmutable<FeedPage>(path);
}

/** Fetches every feed page listed in the manifest and flattens them into a
 * single importance-ordered array. The bundle's own pagination is a
 * transport concern (keeps individual files small and content-addressed);
 * the client re-paginates for display after personalization re-ranks. */
export async function getAllStories(manifest: Manifest): Promise<FeedStoryRow[]> {
  const pages = await Promise.all(
    manifest.pages
      .slice()
      .sort((a, b) => a.page - b.page)
      .map((p) => getFeedPage(p.path)),
  );
  return pages.flatMap((p) => p.stories);
}

export async function getStoryDetail(path: string): Promise<StoryDetail> {
  return fetchImmutable<StoryDetail>(path);
}

export async function getSources(manifest: Manifest): Promise<SourcesFile> {
  // sources.json is rewritten in place each run (not content-addressed —
  // see manifest.sources_path), so it needs the same freshness treatment
  // as the manifest itself.
  return fetchFresh<SourcesFile>(manifest.sources_path);
}

/** Loads the embeddings .bin and returns a lookup from story id to its
 * 384-float vector (a view into one shared Float32Array — no per-story
 * allocation). The path is content-addressed (embeddings_path embeds
 * embeddings_hash), so this is fetched without cache-busting, same as
 * fetchImmutable above. */
export async function getEmbeddingsIndex(
  manifest: Manifest,
): Promise<Map<number, Float32Array>> {
  const url = resolveUrl(manifest.embeddings_path);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const buf = await res.arrayBuffer();
  const dims = manifest.embedding_dimensions;
  const floats = new Float32Array(buf);
  const map = new Map<number, Float32Array>();
  manifest.embeddings_index.forEach((storyId, i) => {
    map.set(storyId, floats.subarray(i * dims, i * dims + dims));
  });
  return map;
}
