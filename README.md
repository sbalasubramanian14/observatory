# Observatory

A personal AI-news reader that collapses duplicate coverage into single stories, ranks
them by importance, and explains why each one matters.

**Live:** <https://the-ai-observatory.vercel.app> · installable as a PWA, reads offline.

---

## The idea

Most feed readers show you articles. Observatory shows you **events**.

When a lab ships a model, forty outlets write about it. A conventional reader gives you
forty rows. Observatory clusters them into one story with the coverage collapsed
underneath — so "how many independent outlets picked this up" becomes a *signal* rather
than noise, and an LLM writes one good summary instead of forty mediocre ones.

That single decision — the unit of the feed is the event, not the article — is what makes
the rest affordable. Roughly 1,700 items a day become ~350 summarization calls.

## How it works

```
sources → COLLECT → NORMALIZE → EMBED → CLUSTER → SCORE → ENRICH → RANK → PUBLISH
          31 feeds   extract     local    multi-    4 free  LLM    Claude  static
          + scrapers  text       model    signal    signals tiers  Code    bundle
```

**SQLite is both the store and the work queue.** Every item carries a `stage` column;
each stage claims rows, processes them, and advances them. No broker, no Redis. It is
resumable after a crash and inspectable with any SQL client.

### Clustering is two steps, not a threshold

A benchmark on real AI headlines found that **embedding similarity alone does not
separate same-story from different-story pairs** — the worst same-story pair scored
*below* the best different-story pair. Cosine similarity by itself is not enough.

Blending it with a crude entity-overlap signal flips the margin positive and widens the
safe threshold band five-fold:

| Signal | Separation margin | Safe threshold band |
|---|---|---|
| cosine only | −0.045 | 0.02 wide |
| entity overlap only | −0.150 | none |
| **0.6·cosine + 0.4·entities** | **+0.004** | **0.10 wide** |

So candidate generation is embedding-based (tuned for recall) and the *decision* is a
blend, with an explicit `AMBIGUOUS` verdict for pairs in the uncertainty band. Ambiguity
resolves to **split, not merge** — a wrong split shows as two visible rows a reader can
reconcile, while a wrong merge silently hides an event.

The regression test asserts the **width of the safe band**, not merely that clustering
works at some threshold. Clustering can produce the right answer while being far too
fragile to ship.

### Three tiers, and they are roles rather than vendors

| Tier | Runs on | Volume | Job |
|---|---|---|---|
| **0** | local, free | ~1,700 items/day | embeddings, entities, dedup, clustering, scoring |
| **1** | free LLM APIs | ~350 stories/day | canonical headline, 2-sentence summary, category |
| **2** | local Claude Code | ~20/day | what is genuinely new, why it matters |

Tier 1 runs through a **priority-ordered failover chain** — Groq → Mistral → OpenRouter →
MiniMax → Gemini. On a real run of 373 stories, Groq exhausted its daily quota after 56 and
auto-disabled; Mistral absorbed 316; **zero stories were lost.** Free tiers have no SLA,
so the answer is several of them rather than a better one.

### Top 50: importance as judged, not as computed

The score below is arithmetic, and that makes it honest about *reach* while
leaving it blind to *meaning* — a heavily syndicated funding round and a frontier
capability result look identical to it. Measured on a live window: the top fifty stories
all scored between 0.50 and 0.52, which is no ranking at all.

So the score only nominates. A shortlist of twice the target size goes to Claude Code in
a single comparative call, which places each story in a band — **landmark** (changes what
is possible or permitted), **significant** (matters to people working in the area),
**notable** (interesting, nothing depends on it) — with one sentence saying why. `/top`
renders those groups.

Ranks are rewritten wholesale each run, so the list rotates instead of accumulating. A
failed ranking writes nothing at all: the site keeps yesterday's judgement, because stale
judgement beats an empty page.

### Importance is reader-independent

The published score is `authority + velocity + novelty`, where **velocity counts distinct
outlets, not articles** — otherwise one publisher's syndication network manufactures
importance.

Personalization happens entirely **on your device**. The bundle ships each story's
embedding; the client keeps positive/negative centroids in `localStorage` and re-ranks
locally. Your interests and reading history never leave your browser — and there is a
toggle to switch it off entirely, because a ranker you cannot audit is not one you should
trust.

### Publishing

The pipeline writes a **content-addressed static bundle** — hashed filenames that are
immutable by construction — and commits it to a separate public data repo. The web client
is deployed once and fetches it at runtime, so **new data needs no redeploy**. It also
makes the service worker's caching trivially correct: cache hashed files forever,
revalidate only the manifest.

The bundle carries a **rolling window** of news (currently 2 days), so the feed stays about what is
happening rather than what has happened. That number is `[publish].retention_days`, and
`observatory.bat 7` overrides it for a single run. Narrowing it destroys nothing: the
database keeps every story regardless, so widening the window and re-publishing brings
the older ones straight back.

Because the window means saved stories eventually leave the bundle, saving keeps its own
copy of the card on the device — otherwise the bookmark button would be a promise the app
could not keep past the end of the week.

Article text is never published. Only titles, canonical links, metadata and
Observatory's own generated summaries — republishing publishers' article bodies would be
copyright redistribution, and that exclusion is enforced in code and tested.

## Coverage

31 sources across four territories: **research** (13), **industry** (10),
**infrastructure** (5), **policy** (3) — arXiv, Hacker News, lab blogs, news outlets,
policy bodies, and hardware/datacenter press. Adding one is editing
`sources.catalogue.toml` and running `feed sources sync`. Feedless sites are handled by a
config-driven `ScraperSource` (CSS selectors, `robots.txt` respected).

Broken connectors are surfaced on a **health page**, because silent coverage loss is the
failure mode that defeats the whole product.

**A source can also be silently empty without being broken.** Collect only asks for items
newer than `now - max_backfill_days`, and `last_run_at` advances on every success, so the
window only ever moves forward. A publisher that posts less often than that cap is
invisible forever — each run, its newest post already predates the window. Nine sources,
Anthropic among them, had contributed exactly zero items while every one of their
connectors worked correctly. `feed sources backfill --days 120` is the way back; it
ignores both the cap and the cadence gate.

## Running it

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

.venv/Scripts/python.exe -m feed doctor          # preflight: providers, auth, disk
.venv/Scripts/python.exe -m feed init
.venv/Scripts/python.exe -m feed sources sync    # load the catalogue
.venv/Scripts/python.exe -m feed run             # collect → score
.venv/Scripts/python.exe -m feed enrich          # LLM tiers
.venv/Scripts/python.exe -m feed publish --out public
```

Or `observatory.bat` for the whole chain including publishing:

```
observatory.bat            full run; window from feed.toml (currently 2 days)
observatory.bat 7          full run; publish only the last 7 days
observatory.bat 7 dryrun   print the command it would run, and stop
```

Repairing a source that publishes less often than the collect cap:

```
feed sources backfill --days 120           # every source
feed sources backfill --days 120 --id anthropic
feed run                                   # push the new items through
```

Provider keys go in a gitignored `.env` (`GROQ_API_KEY`, `MISTRAL_API_KEY`,
`OPENROUTER_API_KEY`, `MINIMAX_API_KEY`, `GEMINI_API_KEY`). All have usable free tiers.

The web client is `web/` — Next.js static export, no server.

## Design notes

The full design document is in [`docs/superpowers/specs/`](docs/superpowers/specs/). It is
worth reading if you are curious about the reasoning: what was deliberately *not* built,
why personalization is a dumb centroid rather than a learned ranker, and why the
uncertainty band exists.

Python: 442 tests. Web: typecheck, lint (including a rule that rejects hardcoded colours
so theming stays token-based), and Playwright end-to-end.

## Status

Working and running. Known gaps, kept honest rather than hidden:

- **Clustering thresholds come from a small hand-labelled corpus.** Genuine duplicate
  coverage only appears around breaking events, so real calibration needs days of
  continuous operation, not a single crawl.
- **Some publishers block image fetching** (403/202), so those stories are text-only.
- **Mobile Lighthouse performance is ~57.** The cause is architectural: fetching data at
  runtime means the hero image is undiscoverable until JS runs. Fixing it fully would
  trade away the no-redeploy-on-publish property.
- **The entity-weight scoring signal is zero-weighted** because nothing persists entity
  rows yet — deliberately switched off rather than left as a constant skewing the
  distribution.

## Licence

Not currently licensed for reuse. Ask if you want to.
