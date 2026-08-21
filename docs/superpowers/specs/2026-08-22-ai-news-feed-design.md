# AI News Feed — Design

**Date:** 2026-08-22
**Status:** Approved, pre-implementation

## 1. Purpose

A personal intelligence system for AI news. It ingests broadly across the AI world, collapses duplicate coverage into single events, explains why each event matters to this reader specifically, and delivers that through a web app, a mobile app, push alerts, and a twice-daily digest.

### Success criteria

1. Nothing important is missed. Broad ingestion, no silent coverage loss.
2. One event produces one row, not forty.
3. A story can be judged in five seconds without opening it.
4. Push notifications stay trustworthy enough to leave enabled.

### Non-goals

- Not a public product at v1. Single user, no signup, no moderation, no billing.
- Not a live streaming terminal. Feed freshness of ~15–30 minutes is sufficient.
- Not an app-store release. Mobile runs as a dev build on the owner's device.
- Not a general reader. AI subject matter only.

## 2. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Audience | Personal, extensible to multi-user | `user_id` present in schema from day one; no auth UI built |
| Rhythm | Fresh feed + rare push + 2x daily digest | Three views over one scored store; no streaming layer needed |
| Coverage | Research, industry, policy, infrastructure | Full breadth, ~600–1200 items/day admitted |
| Feed unit | Story cluster | Dedup becomes the data model; one LLM analysis per event, not per article |
| Runtime | Local-first | Runs on the owner's machine; cloud is a later, separate decision |
| Backend | Python 3.14 | Ingestion and text-extraction ecosystem; 3.15 lacks torch wheels |
| Clients | Next.js (web) + Expo (mobile) | Shared generated types, separate UI |
| LLM providers | Gemini API + local Claude Code | Opposite cost shapes map onto the two processing tiers |
| Extensibility | Plugin protocols for sources, providers, scorers, embedders | Stated hard requirement |

## 3. Architecture

### 3.1 Pipeline

```
sources -> COLLECT -> NORMALIZE -> EMBED -> CLUSTER -> SCORE -> ENRICH -> api
           (plugin)   (extract     (local   (multi-    (signals) (LLM,
                       + canon)     model)   signal)              tiered)
```

**SQLite is both the store and the queue.** Every item carries a `stage` column; each stage is a worker that claims rows (`WHERE stage=? LIMIT n`), processes, and advances them. No broker, no Redis, no Celery. Resumable after crash, inspectable with any SQL client, zero infrastructure. SQLAlchemy keeps a Postgres migration to a connection-string change.

**Stages:**

- **Collect** — source plugins emit `RawItem` on independent cadences.
- **Normalize** — canonical URL, full-text extraction, exact-dupe kill by content hash.
- **Embed** — local embedding model; see 3.3.
- **Cluster** — multi-signal assignment to a story; see 3.4.
- **Score** — five cheap signals, no LLM; see 3.6.
- **Enrich** — the only paid stage; operates on stories, never items. See 3.5.

**Failure isolation is a requirement, not a nicety.** A stage catches per-row, records the error on that row, and continues. With 30+ sources one is always broken; a single bad plugin must never stall the feed.

### 3.2 Data model

- `source` — plugin id, config, cadence, enabled, health
- `item` — raw + normalized fields, url_hash, content_hash, source_id, published_at, text, embedding, embedding_model_id, story_id, stage, error
- `story` — canonical title, kind, first_seen, **updated_at**, **item_count**, score, summary, analysis, analysis_provider, status
- `entity` + `story_entity` — normalized orgs, models, people, papers
- `read_state` — user_id, story_id, seen_at, opened_at, dwell_ms, dismissed
- `edition` — a generated digest, with its story set frozen at generation time

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

`auto` resolves to CUDA when present, otherwise falls back to the ONNX/MiniLM combination, which is the fastest CPU configuration measured. Neither profile is a degraded mode. See Appendix A for the full benchmark.

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
2. **Adjudication** — candidates are scored on a blend of cosine, entity overlap, shared outbound links, and time proximity. Pairs that remain ambiguous after blending are escalated to the Tier 1 LLM answering exactly one question: same event, yes or no.

Embeddings are the recall mechanism, not the decision-maker. Only ambiguous pairs ever cost money.

### 3.5 LLM tiering

- **Tier 0 — free, local.** Embedding, entity extraction, dedup, clustering, scoring. Processes all ~1,000 daily items. No LLM.
- **Tier 1 — bulk, cheap (Gemini Flash).** Once per *story*. Emits canonical headline, two-sentence summary, category, normalized entities. ~30–80 calls/day.
- **Tier 2 — deep, scarce (local Claude Code).** Only above a score cut, ~10–20/day. Emits what is genuinely new versus prior art, what it affects in the reader's work, and connections to already-read stories. Budgeted by call count, not tokens, because it is subscription-billed.

**Provider protocol:**

```python
class Provider(Protocol):
    name: str
    tier: Tier                       # BULK | DEEP
    def complete(self, prompt: str, *, schema: type[T] | None = None) -> T: ...
    def health(self) -> ProviderHealth: ...
```

The router selects by requested tier and **only ever degrades downward**. If the DEEP provider is rate-limited, the story falls back to its Tier 1 summary and is flagged for retry; it never blocks the feed and never silently upgrades. Every analysis records the provider and model that produced it.

### 3.6 Scoring

One weighted sum of five independent Tier-0 signals:

1. **Source authority** — static per-source weight.
2. **Cross-source velocity** — count of *independent outlets* within N hours. Counting outlets rather than articles defeats syndication networks.
3. **Novelty** — max similarity against the last 90 days. High similarity means follow-up, not news.
4. **Entity weight** — importance of involved orgs/models, learned from reading.
5. **Personal fit** — closeness to the interest profile (3.7).

That single score drives all three delivery modes: push above a high cut, digest above a medium cut, feed ranked by it. One number, three consumers.

Each signal is a pluggable function; weights are configuration.

### 3.7 Personalization

Three inputs, in this order:

1. **Written profile.** A prose paragraph describing interests, embedded into a profile vector. Solves cold start on day one with no training data.
2. **Implicit signals.** Opened, dwell time, saved, **dismissed**. Dismissals carry the most information and are usually discarded by such systems.
3. **Explicit corrections.** A more/less-like-this control.

**Mechanism, deliberately simple:** maintain a positive and a negative centroid in embedding space, nudged on interaction. Fit is `cos(story, positive) - cos(story, negative)`. No training, no learned ranker.

Rationale: a single user will never produce the label volume to justify a learned ranker, and the characteristic failure — a feed that quietly hides things and cannot explain itself — is what kills these projects. This is fully inspectable and instantly revertible.

**Serendipity reservation.** A fixed 15% of the feed is reserved for high-importance, low-personal-fit stories, visibly labelled. A perfectly personalized feed guarantees the reader misses things, which directly contradicts success criterion 1.

## 4. Clients

### 4.1 API contract

FastAPI with Pydantic models as the single source of truth, emitting an OpenAPI schema from which TypeScript types are generated. The contract is generated, never hand-maintained, which removes the main cost of a split stack.

```
GET  /feed?cursor=&limit=       ranked stories
GET  /story/{id}                story + evidence items + analysis
POST /story/{id}/interaction    opened | dismissed | saved
GET  /edition/latest            the digest
GET  /search?q=                 semantic + keyword over the archive
GET  /health/sources            connector health
```

`/health/sources` is load-bearing: silent coverage loss is the failure mode that would defeat the whole product.

### 4.2 Web and mobile

- **Web** — Next.js, server-rendered feed read.
- **Mobile** — Expo dev build on the owner's device. No app store at v1.

The clients share the generated API types and data layer. They **do not share UI components**; React and React Native component sharing is a known tar pit and these are small screens.

### 4.3 Push

Delivered via Expo's push service, avoiding direct APNs/FCM work.

The engineering problem is trust, not plumbing: two bad notifications and the feature is dead. Rules: fire only above the high cut; hard cap 2–3/day with cooldown; never push a story whose cluster has already been read.

**Shadow mode is required before enabling.** For the first week the system logs every notification it would have sent and sends none. The threshold is then set against observed data rather than guessed.

### 4.4 Digest

An `edition` is a stored row, not a live query: everything above the medium cut since the previous edition, grouped by category, generated twice daily. Being a stored artifact makes it stable and linkable, and it does not reshuffle while being read.

## 5. Scheduling

The pipeline runs as `python -m feed.pipeline run`, a plain CLI. Locally that is Windows Task Scheduler or in-process APScheduler; on a VPS it is cron or a systemd timer. The pipeline has no knowledge of what triggered it, which keeps the local-first-then-maybe-cloud path open.

## 6. Testing

Pipeline stages are pure functions over rows and unit-test with fixtures.

**Clustering requires a golden-set regression test.** A hand-labelled corpus (seeded from the 22-item spike corpus) is asserted on every change to clustering signals, checking that the safe-threshold band does not narrow. The measured band is 0.02 wide with cosine alone and 0.10 blended; a regression here would silently wreck the feed and no other test would catch it.

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

The system is built in four phases, each independently useful. Later phases must not be started before the preceding one runs on real data.

1. **Pipeline to stored stories.** Collect through Score, plus 5–10 sources and the CLI. Output is inspected in SQL. This phase is where clustering thresholds get derived from real data, replacing the directional values in Appendix A.
2. **Enrichment and API.** Tier 1 and Tier 2 providers, the router, and the six endpoints. Output is inspected as JSON.
3. **Web client and digest.** The first phase with a human-facing surface.
4. **Mobile client and push.** Includes the mandatory shadow-mode week before notifications are enabled.

A good pipeline with an ugly web page is useful on day one; a polished app over a mediocre feed is worthless. Phase order reflects that.

## 9. Open questions

1. Tier 2 daily budget — starts at 20 stories/day, to be tuned against observed Claude Code rate limits.
2. Serendipity share — starts at 15%, to be tuned by feel after a week.
3. Initial source list and per-source authority weights — deferred to the implementation plan.
4. Clustering time-window width — a starting value must be derived from real data during phase 1.
5. Score cut points for push, digest, and Tier 2 eligibility — three thresholds on the single score, all deliberately unset. They cannot be guessed and must come from a live score distribution observed in phase 1; the push cut is additionally validated by the shadow-mode week in phase 4.
6. Signal weights in the scoring sum — starting values to be set in phase 1 and revised once a score distribution exists.
