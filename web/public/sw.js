// Observatory service worker.
//
// Caching strategy (see docs/superpowers/specs/2026-08-22-ai-news-feed-design.md
// §4.1 — the bundle is content-addressed: hashed filenames like
// feed/page-0-<hash>.json never change body once published):
//
//   1. Hashed bundle files (feed/*, story/*, embeddings/*.bin) -> cache-first,
//      effectively permanent. A hit never re-validates the network at all —
//      the hash IS the validation.
//   2. manifest.json / sources.json -> network-first, always revalidated.
//      These are overwritten in place every pipeline run and are already
//      cache-busted by the client (see src/lib/bundle.ts fetchFresh), so the
//      service worker's job is just to have a cached fallback ready for when
//      the network is unavailable, never to prefer its own cached copy.
//   3. The app shell (the static-export HTML/JS/CSS Next.js emits, plus the
//      external bundle host's own document) -> stale-while-revalidate: serve
//      the cache immediately for a fast/offline load, then refresh it in the
//      background for next time.
//
// Two separate cache buckets so an app-shell update (new deploy) never
// evicts still-good hashed data, and a bundle republish never evicts the
// app shell:
const SHELL_CACHE = "observatory-shell-v1";
const IMMUTABLE_CACHE = "observatory-immutable-v1";
const FRESH_CACHE = "observatory-fresh-v1";
const ALL_CACHES = [SHELL_CACHE, IMMUTABLE_CACHE, FRESH_CACHE];

// Every top-level route's HTML shell, precached rather than relying on
// the reader having hard-navigated to it before -- Next's client router
// uses small RSC "flight" payload fetches for in-app Link clicks (see
// fetch handler below), which never touch the full HTML document, so a
// route a reader only ever *soft*-navigated to would otherwise have no
// offline-servable document at all the first time they land on it
// directly (a deep link, a PWA relaunch, a hard reload).
//
// Split into two tiers on purpose: CRITICAL_SHELL_URLS blocks `install`
// (a reader who reloads *this* page in the next few seconds should find
// it instantly offline-capable); DEFERRED_SHELL_URLS is everything else,
// started only after activation so it never competes with the current
// page's own critical-path fetches (manifest.json, feed pages, lead
// images) for the same throttled connection -- precaching six routes'
// worth of HTML plus every JS/CSS/font chunk they reference is real
// bandwidth, and doing it eagerly during install measurably pushed out
// first-load LCP.
const CRITICAL_SHELL_URLS = ["/", "/manifest.webmanifest"];
const DEFERRED_SHELL_URLS = ["/story/", "/saved/", "/dismissed/", "/sources/"];

// The JS/CSS/font chunks a page needs can never be intercepted (and thus
// opportunistically cached) on the very load that registers this worker --
// registering it is itself part of that bundle, so it doesn't exist yet
// when those requests go out. Without precaching them explicitly, the
// *first-ever* visit would leave those files un-cached, and offline would
// only ever get as far as an unstyled, unhydrated shell (the HTML loads
// from SHELL_CACHE, but every <script>/<link> it references 404s against
// the network). Their filenames are content-hashed per build, so instead
// of a hardcoded list this parses each precached HTML page for its own
// /_next/static/... references and precaches the union of those too.
const ASSET_URL_RE = /(?:src|href)="(\/_next\/static\/[^"]+\.(?:js|css|woff2?))"/g;

async function precacheShell(cache, urls) {
  const assetUrls = new Set();
  await Promise.all(
    urls.map(async (url) => {
      try {
        const res = await fetch(url);
        if (!res.ok) return;
        const html = await res.clone().text();
        await cache.put(url, res);
        for (const match of html.matchAll(ASSET_URL_RE)) {
          assetUrls.add(new URL(match[1], self.location.origin).toString());
        }
      } catch {
        // A single route failing to precache (e.g. install while offline)
        // shouldn't block the rest — the fetch handler still caches
        // opportunistically as those routes are actually visited.
      }
    }),
  );
  await Promise.all(
    Array.from(assetUrls).map(async (assetUrl) => {
      try {
        const cached = await cache.match(assetUrl);
        if (cached) return; // already have it from an earlier precache pass
        const res = await fetch(assetUrl);
        if (res.ok) await cache.put(assetUrl, res);
      } catch {
        // same rationale as above
      }
    }),
  );
}

function precacheDeferred() {
  caches.open(SHELL_CACHE).then((cache) => precacheShell(cache, DEFERRED_SHELL_URLS));
}

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => precacheShell(cache, CRITICAL_SHELL_URLS)));
  // Do NOT self.skipWaiting() unconditionally here — the client controls
  // the moment of activation (see the SKIP_WAITING message handler) so an
  // update never yanks the page out from under mid-read state without the
  // reader being told (UpdateToast.tsx).
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((key) => !ALL_CACHES.includes(key)).map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim())
      .then(() => {
        // Fire-and-forget, after activation has resolved (not blocking
        // it) and on a short delay -- long enough that it lands after the
        // controlled page's own first-load fetches have had the
        // connection to themselves, short enough that it's still done
        // within the same "one visit" a reader who then goes offline
        // would expect full coverage from.
        setTimeout(precacheDeferred, 4000);
      }),
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") self.skipWaiting();
});

function isImmutableBundlePath(url) {
  // Content-addressed: .../feed/page-N-<hash>.json, .../story/<id>-<hash>.json,
  // .../embeddings/<date>-<hash>.bin — every one carries a hash segment in
  // its filename, which is the actual invariant (not "which folder"), but
  // matching the folder names is enough here and keeps the regex readable.
  return /\/(feed|story|embeddings)\/[^/]+\.(json|bin)$/.test(url.pathname);
}

function isFreshBundlePath(url) {
  return /\/(manifest|sources)\.json$/.test(url.pathname);
}

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (err) {
    // ignoreSearch: this request carries a cache-busting ?t=<timestamp>
    // query (see lib/bundle.ts's fetchFresh) that's different on every
    // call by design, so an exact-URL cache lookup would never match a
    // previously cached response and this fallback would never fire.
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) {
      // A network failure served from cache is NOT the same event as a
      // genuine fresh fetch — the client (lib/pwa.ts's recordSync, called
      // from lib/bundle.ts) must be able to tell the difference so the
      // offline banner's "cached data from <time>" timestamp stays honest
      // instead of silently rewriting itself to "now" on every offline
      // reload. This header is that signal; the browser's own fetch()
      // can't otherwise distinguish a service-worker cache hit from a
      // real network response.
      const headers = new Headers(cached.headers);
      headers.set("X-Observatory-From-Cache", "1");
      return new Response(cached.body, {
        status: cached.status,
        statusText: cached.statusText,
        headers,
      });
    }
    throw err;
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  // ignoreSearch: this is the app-shell HTML for a statically-exported
  // client-routed page (e.g. /story/), which is byte-identical regardless
  // of its query string (?id=397 vs ?id=402) -- the id is read client-side
  // via useSearchParams after hydration, not baked into the file. Without
  // this, visiting story 397 online would never satisfy an offline
  // navigation to story 402's URL even though it's the exact same HTML.
  const cached = await cache.match(request, { ignoreSearch: true });
  const network = fetch(request)
    .then((response) => {
      if (response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => undefined);
  return cached || (await network) || Response.error();
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Cross-origin bundle files (raw.githubusercontent.com / whatever
  // NEXT_PUBLIC_BUNDLE_URL points at) still go through the same rules —
  // the content-addressing property holds regardless of origin. Opaque
  // cross-origin responses can still be cached and replayed even though
  // their status can't be inspected, so `response.ok` checks above are
  // skipped for those by just trusting a 0-status opaque response too.
  if (isImmutableBundlePath(url)) {
    event.respondWith(cacheFirstOpaque(request, IMMUTABLE_CACHE));
    return;
  }
  if (isFreshBundlePath(url)) {
    event.respondWith(networkFirst(request, FRESH_CACHE));
    return;
  }
  if (url.origin === self.location.origin) {
    event.respondWith(staleWhileRevalidate(request, SHELL_CACHE));
  }
  // Everything else (analytics-less third parties, if any) passes through
  // untouched — no respondWith call means the browser's default handling.
});

async function cacheFirstOpaque(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  // An opaque (cross-origin, no-cors) response has status 0 and can't be
  // inspected for .ok, but it's still perfectly cacheable and replayable —
  // the browser handles decoding it as if it were the real response when
  // it's later returned from the Cache API.
  if (response && (response.ok || response.type === "opaque")) {
    const cache = await caches.open(cacheName);
    cache.put(request, response.clone());
  }
  return response;
}
