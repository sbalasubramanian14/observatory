// Fetches the published bundle at runtime (never bundled at build time —
// spec §4.5: "Deployed once; it fetches the manifest and bundle at
// runtime"). All paths are root-relative by default so they work against
// the locally-served copy in web/public/data; set
// NEXT_PUBLIC_BUNDLE_BASE_URL to point the same client code at a CDN
// without a rebuild.
import type {
  FeedPage,
  FeedStoryRow,
  Manifest,
  SourcesFile,
  StoryDetail,
} from "./types";

const DATA_BASE = (
  process.env.NEXT_PUBLIC_BUNDLE_BASE_URL ?? "/data"
).replace(/\/$/, "");

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${DATA_BASE}/${path.replace(/^\//, "")}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function getManifest(): Promise<Manifest> {
  return fetchJson<Manifest>("manifest.json");
}

export async function getFeedPage(path: string): Promise<FeedPage> {
  return fetchJson<FeedPage>(path);
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
  return fetchJson<StoryDetail>(path);
}

export async function getSources(manifest: Manifest): Promise<SourcesFile> {
  return fetchJson<SourcesFile>(manifest.sources_path);
}

/** Loads the embeddings .bin and returns a lookup from story id to its
 * 384-float vector (a view into one shared Float32Array — no per-story
 * allocation). */
export async function getEmbeddingsIndex(
  manifest: Manifest,
): Promise<Map<number, Float32Array>> {
  const url = `${DATA_BASE}/${manifest.embeddings_path.replace(/^\//, "")}`;
  const res = await fetch(url, { cache: "no-store" });
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
