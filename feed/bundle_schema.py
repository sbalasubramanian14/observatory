from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    # extra="forbid" is the enforcement mechanism for spec 4.2's hard
    # exclusions: if anything upstream ever starts stuffing item.text (or
    # anything else not modelled here) into a payload dict, model_validate()
    # raises instead of silently publishing it. This is the "bundle schema
    # contract test" spec 6 requires -- the Publish stage validates against
    # this schema before writing anything to disk, and refuses to publish
    # on failure.
    model_config = ConfigDict(extra="forbid")


class StoryEvidence(_Strict):
    """One contributing article. Titles, canonical links, and metadata
    only -- NEVER item.text (spec 4.2: publishing scraped article bodies is
    copyright redistribution) and never item.summary (an unvetted
    publisher-supplied blurb, not "our own generated summary")."""
    id: int
    title: str
    url: str
    source_id: str
    published_at: str | None = None


class StoryDetail(_Strict):
    """story/<id>-<hash>.json -- evidence links plus our own analysis."""
    id: int
    title: str
    kind: str | None = None
    category: str | None = None
    summary: str | None = None
    summary_provider: str | None = None
    analysis: str | None = None
    analysis_provider: str | None = None
    score: float | None = None
    score_breakdown: dict | None = None
    first_seen: str
    updated_at: str
    item_count: int
    outlet_count: int
    # spec D0: the story's lead image URL, picked from its highest-authority
    # contributing item (see feed.stages.publish._lead_image_for). A URL
    # reference, never image bytes -- spec 4.2's "no full article text"
    # exclusion is about redistributing copyrighted body text, not linking
    # to a publisher-hosted image, so this does not violate it. None is the
    # common case (most stories have no usable image) and clients must
    # render that gracefully, not as an error.
    lead_image_url: str | None = None
    evidence: list[StoryEvidence]


class FeedPageStory(_Strict):
    """One row in a feed page: enough to render a card without a second
    fetch, plus the path to the full detail file."""
    id: int
    title: str
    kind: str | None = None
    category: str | None = None
    summary: str | None = None
    score: float | None = None
    item_count: int
    outlet_count: int
    updated_at: str
    detail_path: str
    detail_hash: str
    # See StoryDetail.lead_image_url -- carried here too so the feed card
    # can render an image without a second fetch to the detail file.
    lead_image_url: str | None = None
    # Web source/territory filter: the distinct set of source ids across
    # this story's contributing items, sorted for a stable, diffable
    # bundle. A story clusters items from possibly-multiple sources (spec
    # A1's whole point), and territory itself only lives per-source
    # (sources.json) -- so a client-side filter needs this list to know
    # whether ANY of a story's items match a selected source or territory,
    # without fetching every story's full detail file just to filter the
    # main list. Never empty for a story that made it through cluster().
    source_ids: list[str]
    # Top 50 -- importance as JUDGED by the DEEP provider, as opposed to
    # the arithmetic `score` above (feed/stages/rank.py explains why the
    # two disagree). All three are null together for a story outside the
    # current Top N, which is nearly all of them.
    importance_rank: int | None = None
    importance_band: str | None = None
    importance_reason: str | None = None


class FeedPage(_Strict):
    """feed/page-<n>-<hash>.json -- importance-ranked stories, paginated."""
    page: int
    page_count: int
    stories: list[FeedPageStory]


class ManifestPage(_Strict):
    page: int
    path: str
    hash: str
    count: int


class Manifest(_Strict):
    """manifest.json -- small, always refetched. Every other file's name
    carries a content hash; only this one is refetched by stable name, so
    it is the map clients use to discover everything else."""
    version: int
    generated_at: str
    embedding_model_id: str | None
    embedding_dimensions: int | None
    story_count: int
    pages: list[ManifestPage]
    embeddings_path: str | None
    embeddings_hash: str | None
    embeddings_index: list[int]
    sources_path: str
    retention_days: int


class SourceHealth(_Strict):
    id: str
    plugin: str
    enabled: bool
    cadence_minutes: int
    last_run_at: str | None = None
    consecutive_failures: int
    last_error: str | None = None
    # Spec A3/A4: set when the most recent collect() run has a specific
    # reason to suspect coverage loss (RSS window truncation, or the
    # backfill cap narrowing the fetch window) -- None means the last run
    # collected everything since the previous one, per source.
    coverage_warning: str | None = None
    # Spec 2's four coverage territories (research | industry | policy |
    # infrastructure), from sources.catalogue.toml via `feed sources sync`.
    # None for a source never synced from the catalogue.
    territory: str | None = None


class SourcesReport(_Strict):
    """sources.json -- connector health and coverage report. Published
    deliberately (spec 4.2): silent coverage loss is the failure mode that
    would defeat success criterion 1, and it must be visible in the
    client."""
    generated_at: str
    sources: list[SourceHealth]
