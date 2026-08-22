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
    analysis: str | None = None
    analysis_provider: str | None = None
    score: float | None = None
    score_breakdown: dict | None = None
    first_seen: str
    updated_at: str
    item_count: int
    outlet_count: int
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


class SourcesReport(_Strict):
    """sources.json -- connector health and coverage report. Published
    deliberately (spec 4.2): silent coverage loss is the failure mode that
    would defeat success criterion 1, and it must be visible in the
    client."""
    generated_at: str
    sources: list[SourceHealth]
