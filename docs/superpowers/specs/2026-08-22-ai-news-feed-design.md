# AI News Feed — Design

**Date:** 2026-08-22
**Status:** Approved, pre-implementation

## 1. Purpose

A personal intelligence system for AI news. It ingests broadly across the AI world, collapses duplicate coverage into single events, explains why each event matters, and delivers that through a web app, a mobile app, and a twice-daily digest.

### Success criteria

1. Nothing important is missed. Broad ingestion, no silent coverage loss.
2. One event produces one row, not forty.
3. A story can be judged in five seconds without opening it.
4. Readable on the phone at 8am on cellular with the laptop shut.

### Non-goals

- Not a public product at v1. Single reader, no signup, no moderation, no billing.
- Not a live streaming terminal. Feed freshness of ~15–30 minutes is sufficient.
- **No push notifications.** Explicitly dropped. Reading is pull-only.
- Not an app-store release. Mobile runs as a dev build on the owner's device.
- Not a general reader. AI subject matter only.

## 2. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Audience | Personal, extensible to multi-user | `user_id` present in schema from day one; no auth UI built |
| Rhythm | Fresh feed + 2x daily digest | Two views over one scored store; no streaming, no push |
| Coverage | Research, industry, policy, infrastructure | Full breadth, ~600–1200 items/day admitted |
| Feed unit | Story cluster | Dedup becomes the data model; one LLM analysis per event, not per article |
| Runtime | Local-first | Pipeline runs on the owner's machine |
| **Serving** | **Published static bundle** | **No API server. Pipeline commits JSON to a public repo; clients read it directly** |
| Backend | Python 3.14 | Ingestion and text-extraction ecosystem; 3.15 lacks torch wheels |
| Web | Next.js static, hosted on Vercel | Deployed once; data fetched at runtime, so data updates need no rebuild |
| Mobile | Expo, device SQLite cache | True offline reading, not merely online-with-cache |
| LLM providers | Gemini API + local Claude Code | Opposite cost shapes map onto the two processing tiers |
| Extensibility | Plugin protocols for sources, providers, scorers, embedders | Stated hard requirement |

## 3. Architecture

### 3.1 Pipeline

```
sources -> COLLECT -> NORMALIZE -> EMBED -> CLUSTER -> SCORE -> ENRICH -> PUBLISH
           (plugin)   (extract     (local   (multi-    (signals) (LLM,      (static
                       + canon)     model)   signal)              tiered)    bundle)
```

**SQLite is both the store and the queue.** Every item carries a `stage` column; each stage is a worker that claims rows (`WHERE stage=? LIMIT n`), processes, and advances them. No broker, no Redis, no Celery. Resumable after crash, inspectable with any SQL client, zero infrastructure.

**Stages:**

- **Collect** — source plugins emit `RawItem` on independent cadences.
- **Normalize** — canonical URL, full-text extraction, exact-dupe kill by content hash.
- **Embed** — local embedding model; see 3.3.
- **Cluster** — multi-signal assignment to a story; see 3.4.
- **Score** — five cheap signals, no LLM; see 3.6.
- **Enrich** — the only paid stage; operates on stories, never items. See 3.5.
- **Publish** — emits the static bundle and commits it; see 4.

**Failure isolation is a requirement, not a nicety.** A stage catches per-row, records the error on that row, and continues. With 30+ sources one is always broken; a single bad plugin must never stall the feed.

### 3.2 Data model

Local SQLite, never published:

- `source` — plugin id, config, cadence, enabled, health
- `item` — raw + normalized fields, url_hash, content_hash, source_id, published_at, **full text**, embedding, embedding_model_id, story_id, stage, error
- `story` — canonical title, kind, first_seen, **updated_at**, **item_count**, score, summary, analysis, analysis_provider, status
- `entity` + `story_entity` — normalized orgs, models, people, papers
- `edition` — a generated digest, story set frozen at generation time

Read state lives **on each device**, not here. See 4.3.

`story.updated_at` and `item_count` exist so living threads (stories that develop over days, showing only what changed) become a query later rather than a migration.

### 3.3 Embedding

Vectors from different models are not comparable and their thresholds differ. Therefore **every stored vector records its `embedding_model_id`**, and changing models triggers a corpus re-embed rather than silently corrupting clusters.

Two first-class profiles, one switch:

```toml
[embedding]
backend    = "torch"                 # torch | onnx
model      = "BAAI/bge-small-en-v1.5"
device     = "auto"                  # auto | cuda | cpu
batch_size = 256
```

- **Workstation** — torch + CUDA + bge-small-en-v1.5. ~203 docs/s measured.
- **Server** (no GPU) — onnx + CPU + all-MiniLM-L6-v2. ~90 docs/s measured.

`auto` resolves to CUDA when present, otherwise the ONNX/MiniLM combination, which is the fastest CPU configuration measured. Neither profile is a degraded mode. See Appendix A.

Note: the published bundle carries story embeddings, so the model choice is also a **client-visible** contract — the manifest records `embedding_model_id` and clients must discard cached centroids when it changes.

### 3.4 Clustering

**Measured finding that shaped this design:** embedding cosine similarity alone does not separate same-story from different-story pairs. On a 22-item labelled corpus both candidate models produced a *negative* margin (worst same-story pair scored below the best different-story pair). Perfect recovery occurred only inside a threshold band 0.02 wide, via transitive chaining — too fragile for production.

Blending cosine with a free entity-overlap signal flipped the margin positive and widened the safe band 5x:

```
cosine only                      margin -0.045   safe band 0.02 wide
entity overlap only              margin -0.150   safe band NONE
0.6*cosine + 0.4*entities        margin +0.004   safe band 0.10 wide
```

**Therefore clustering is two steps, not a threshold:**

1. **Candidate generation** — embeddings retrieve near neighbours within a time window. Tuned for recall, not precision.
2. **Adjudication** — candidates scored on a blend of cosine, entity overlap, shared outbound links, and time proximity. Pairs still ambiguous after blending escalate to the Tier 1 LLM answering one question: same event, yes or no.

Embeddings are the recall mechanism, not the decision-maker. Only ambiguous pairs cost money.

### 3.5 LLM tiering

- **Tier 0 — free, local.** Embedding, entity extraction, dedup, clustering, scoring. Processes all ~1,000 daily items. No LLM.
- **Tier 1 — bulk, cheap (Gemini Flash).** Once per *story*. Emits canonical headline, two-sentence summary, category, normalized entities. ~30–80 calls/day.
- **Tier 2 — deep, scarce (local Claude Code).** Only above a score cut, ~10–20/day. Emits what is genuinely new versus prior art, what it affects, and connections to earlier stories. Budgeted by call count, not tokens, because it is subscription-billed.

**Provider protocol:**

```python
class Provider(Protocol):
    name: str
    tier: Tier                       # BULK | DEEP
    def complete(self, prompt: str, *, schema: type[T] | None = None) -> T: ...
    def health(self) -> ProviderHealth: ...
```

The router selects by requested tier and **only ever degrades downward**. If the DEEP provider is rate-limited, the story falls back to its Tier 1 summary and is flagged for retry; it never blocks the pipeline and never silently upgrades. Every analysis records the provider and model that produced it.

### 3.6 Scoring

One weighted sum of five independent Tier-0 signals, all computed in the pipeline:

1. **Source authority** — static per-source weight.
2. **Cross-source velocity** — count of *independent outlets* within N hours. Counting outlets rather than articles defeats syndication networks.
3. **Novelty** — max similarity against the last 90 days. High similarity means follow-up, not news.
4. **Entity weight** — importance of involved orgs and models.
5. *(Personal fit is deliberately NOT here — see 4.3.)*

This produces an **importance score**, which is reader-independent. It drives digest inclusion and Tier 2 eligibility, and provides the baseline feed order in the bundle.

Each signal is a pluggable function; weights are configuration.

### 3.7 Personalization

Personalization runs **entirely on the client**, because the bundle is public and must carry no reader signal. See 4.3 for the mechanism.

## 4. Serving

There is **no API server**. The pipeline's terminal stage writes a static bundle and commits it to a public Git repository; clients read that directly over a CDN. This removes a server to run, secure, and keep awake, and it means the feed is readable whether or not the laptop exists.

It also removes duplicated work: ranking and rendering happen once, at publish time, rather than being recomputed per request to produce output that is identical until the next pipeline run.

### 4.1 Bundle layout

```
manifest.json                    small, always fetched fresh
feed/page-<n>-<hash>.json        importance-ranked stories, paginated
story/<id>-<hash>.json           story detail: evidence links + analysis
edition/<date>-<slot>.json       digest
embeddings/<window>-<hash>.bin   story vectors for client-side ranking
sources.json                     connector health and coverage report
```

**Manifest plus content-addressed filenames.** Every data file's name contains a content hash, making it immutable and therefore cacheable forever by any CDN. Only `manifest.json` is refetched, and it is small. This solves staleness and caching in one move — no cache purging, no CDN TTL fighting.

### 4.2 What must never enter the bundle

The repo is public. Two hard exclusions:

- **Full article text.** Republishing publishers' article bodies is redistribution of copyrighted work. Full text stays in local SQLite. The bundle carries titles, canonical links, metadata, and the system's own generated summaries and analysis — which is both legally clean and far smaller.
- **Reader behaviour.** Opens, dwell times, dismissals, and the interest profile never leave the device.

`sources.json` is included deliberately: silent coverage loss is the failure mode that would defeat success criterion 1, and it must be visible in the client.

### 4.3 Client-side personalization

The bundle ships stories ranked by **importance only**, plus each story's 384-float embedding (~1.5 KB per story, ~75 KB/day — negligible).

Each client then, locally:

1. Maintains a **positive and a negative centroid** in embedding space, nudged by opens, dwell, saves, and dismissals.
2. Computes fit as `cos(story, positive) - cos(story, negative)`.
3. Re-ranks the importance-ordered feed by a blend of importance and fit.
4. Filters out already-read stories.
5. Reserves **15%** of feed slots for high-importance, low-fit stories, visibly labelled.

This is a dot product over a few hundred vectors — free on any device.

Three consequences, all good: the reader's profile and history never leave their devices; there is no write-back sync to build; and the mechanism stays fully inspectable and instantly revertible. A learned ranker is rejected for the same reason as before — a single reader will never produce the label volume to justify one, and a feed that quietly hides things and cannot explain itself is what kills these projects.

Cold start is solved by a written profile paragraph, embedded on-device into the initial positive centroid.

The serendipity reservation is not optional: a perfectly personalized feed guarantees the reader misses things, contradicting success criterion 1.

### 4.4 Publishing mechanics

The Publish stage writes the bundle, commits, and pushes to the public data repo. Because content-addressed files are immutable, each run adds new files rather than rewriting them.

**Repo growth must be managed from day one.** At roughly 50 stories/day the bundle grows a few hundred KB per day; unmanaged, Git history reaches hundreds of MB within a year. The Publish stage therefore prunes files outside a rolling window (default 90 days) and the repo is periodically history-squashed. Deep archive stays in local SQLite, which is the system of record.

### 4.5 Clients

**Web** — Next.js, statically exported, hosted on Vercel. Deployed **once**; it fetches the manifest and bundle at runtime. Data updates therefore require no rebuild and no redeploy, which also keeps the pipeline's publish cadence independent of Vercel's deployment limits.

**Mobile** — Expo. Fetches the same URLs, mirrors the bundle into device SQLite, and serves entirely from that cache. This is genuine offline reading, not online-with-cache: after one sync the app is fully functional with no network.

Clients share the generated TypeScript types for the bundle schema and the fetch/cache layer. They **do not share UI components**; React and React Native component sharing is a known tar pit and these are small screens.

### 4.6 Digest

An `edition` is a published artifact, not a live query: everything above the medium importance cut since the previous edition, grouped by category, generated twice daily. Being immutable makes it stable and linkable, and it does not reshuffle while being read.

## 5. Scheduling

The pipeline runs as `python -m feed.pipeline run`, a plain CLI. Locally that is Windows Task Scheduler or in-process APScheduler; on a VPS it would be cron or a systemd timer. The pipeline has no knowledge of what triggered it, which keeps the local-first-then-maybe-cloud path open.

## 6. Testing

Pipeline stages are pure functions over rows and unit-test with fixtures.

**Clustering requires a golden-set regression test.** A hand-labelled corpus (seeded from the 22-item spike corpus) is asserted on every change to clustering signals, checking that the safe-threshold band does not narrow. The measured band is 0.02 wide with cosine alone and 0.10 blended; a regression here would silently wreck the feed and no other test would catch it.

**The bundle schema requires a contract test.** Since clients are deployed independently of the pipeline, a published bundle that violates the schema breaks readers with no server to hotfix. The Publish stage validates against the schema before committing, and refuses to publish on failure.

## 7. Appendix A — Measured benchmarks

Hardware: Ryzen 9 7940HS (8C/16T), 15.2 GB RAM, RTX 4050 Laptop (6 GB VRAM), Windows 11, Python 3.14.3. Documents ~1800 chars.

| Model | Runtime | Device | docs/s | 1,000 items |
|---|---|---|---|---|
| bge-small-en-v1.5 | ONNX | CPU | 2.8 | 6 min |
| bge-small-en-v1.5 | PyTorch | CPU | 15.1 | 66 s |
| all-MiniLM-L6-v2 | PyTorch | CPU | 44.4 | 23 s |
| all-MiniLM-L6-v2 | ONNX | CPU | 89.6 | 11 s |
| bge-small-en-v1.5 | PyTorch | CUDA | 203.2 | 4.9 s |
| all-MiniLM-L6-v2 | PyTorch | CUDA | 616.2 | 1.6 s |

Notes:

- fastembed's bge-small ONNX export is anomalously slow (5.4x slower than the same model under PyTorch, against an expected speedup). Its MiniLM export is fine. **fastembed's speed advantage is per-model, not universal.**
- GPU speedup is 13.5–13.9x. Peak VRAM 1.31 GB (MiniLM, batch 256) and 0.82 GB (bge, batch 64) — well within 6 GB.
- Batch 512 *reduced* MiniLM throughput to 447/s on padding waste. 256 is the sweet spot.
- Batch size must be pinned: an unpinned batch over 1,000 docs peaked at 5.6 GB RSS. At batch 64 it sits at 430–570 MB.
- Python 3.15 has no torch wheels. The project pins 3.14.
- Install cost: fastembed 32 s / 45 MB runtime; PyTorch CPU ~3 min / 533 MB; PyTorch CUDA 6m34s.

**Caveats.** The clustering corpus is 22 hand-written items authored by the assistant; results are directional, sufficient to shape the design but not to fix constants. The blend was validated on bge-small only, not MiniLM. Thresholds must be re-derived on real data.

## 8. Build order

Four phases, each independently useful. Later phases must not begin before the preceding one runs on real data.

1. **Pipeline to stored stories.** Collect through Score, plus 5–10 sources and the CLI. Output inspected in SQL. This is where clustering thresholds get derived from real data, replacing the directional values in Appendix A.
2. **Enrichment and publishing.** Tier 1 and Tier 2 providers, the router, the bundle writer, and the push-to-repo mechanics. Output inspected as JSON in the public repo.
3. **Web client.** Next.js reader on Vercel, including client-side ranking and the digest view. First human-facing surface.
4. **Mobile client.** Expo app with device SQLite cache and offline reading.

A good pipeline with an ugly web page is useful on day one; a polished app over a mediocre feed is worthless. Phase order reflects that.

## 9. Open questions

1. **The public data repo** — to be supplied by the owner. Determines bundle base URL and CDN strategy (raw Git host vs jsDelivr vs Vercel-served).
2. Tier 2 daily budget — starts at 20 stories/day, to be tuned against observed Claude Code rate limits.
3. Serendipity share — starts at 15%, to be tuned by feel after a week.
4. Initial source list and per-source authority weights — deferred to the implementation plan.
5. Clustering time-window width — starting value must be derived from real data in phase 1.
6. Score cut points for digest inclusion and Tier 2 eligibility — deliberately unset. They cannot be guessed and must come from a live score distribution observed in phase 1.
7. Signal weights in the scoring sum — starting values set in phase 1, revised once a score distribution exists.
8. Bundle retention window — starts at 90 days; revisit once real bundle sizes are known.
