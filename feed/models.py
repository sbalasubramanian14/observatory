from __future__ import annotations
import enum
from datetime import datetime, timezone
from sqlalchemy import (
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text,
    TypeDecorator, UniqueConstraint, LargeBinary,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


class UtcDateTime(TypeDecorator):
    """DateTime that always round-trips as an aware UTC datetime.

    SQLite has no native timezone-aware datetime type: SQLAlchemy's plain
    DateTime(timezone=True) happily accepts an aware datetime on write but
    hands back a naive one on read, since sqlite just stores an ISO string
    with no offset. That naive/aware mismatch then blows up (or silently
    misbehaves) anywhere a column value is later compared against an aware
    "now". This type normalises both directions so every datetime column
    reads back the same aware UTC value that was written.
    """
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if value.tzinfo is None:
            # Naive input is treated as already-UTC rather than guessed as
            # local time -- callers in this codebase always deal in UTC.
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

class Stage(enum.Enum):
    COLLECTED = "collected"
    NORMALIZED = "normalized"
    EMBEDDED = "embedded"
    CLUSTERED = "clustered"
    SCORED = "scored"
    FAILED = "failed"


class StoryStatus(enum.Enum):
    """Enrich-stage progress for a story, per spec 3.5's LLM tiering.

    NEW       -- scored, not yet through Tier 1.
    ENRICHED  -- Tier 1 (bulk headline/summary/category) done.
    ANALYZED  -- Tier 2 (deep "why this matters") done by the real DEEP
                 provider.
    RETRY     -- Tier 2 was attempted but the DEEP provider was unavailable
                 or failed, so the router degraded to the BULK provider's
                 simpler prompt (spec 3.5: "flagged for retry"). The story
                 keeps a usable analysis in the meantime but stays eligible
                 for re-selection next time Tier 2 runs.
    """
    NEW = "new"
    ENRICHED = "enriched"
    ANALYZED = "analyzed"
    RETRY = "retry"

class Source(Base):
    __tablename__ = "source"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    plugin: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    cadence_minutes: Mapped[int] = mapped_column(Integer, default=30)
    authority: Mapped[float] = mapped_column(Float, default=0.5)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    # Spec A4: per-source override of [collect].max_backfill_days. None
    # means "use the global default" -- see feed.stages.collect._effective_since.
    max_backfill_days: Mapped[int | None] = mapped_column(Integer)
    # Spec A3/A4: the most recent run's coverage-loss signal, refreshed
    # every run (cleared to None on a clean run, same lifecycle as
    # last_error/consecutive_failures above) -- NOT sticky, since it
    # describes "is there a reason to distrust the LAST run's coverage",
    # not a running history. Set either when the backfill cap actually
    # narrowed the fetch window, or when a source plugin's own fetch()
    # reports a coverage_warning attribute (e.g. RSS: every dated entry in
    # the fetched feed postdates `since`, suggesting the feed rolled past
    # older items between runs). Published in sources.json (spec 4.2:
    # "silent coverage loss ... must be visible in the client").
    coverage_warning: Mapped[str | None] = mapped_column(Text)
    # Spec 2's four coverage territories (research | industry | policy |
    # infrastructure), populated from sources.catalogue.toml by `feed
    # sources sync` (see feed.catalogue / feed.stages.sync). Nullable so a
    # pre-existing row added by hand (e.g. `feed sources add`, or a row
    # from before this column existed) does not fail to load -- it just
    # shows as "no territory" in sources.json / the territory-mix report
    # until the next sync assigns one.
    territory: Mapped[str | None] = mapped_column(String(32))

class Item(Base):
    __tablename__ = "item"
    __table_args__ = (UniqueConstraint("url_hash", name="uq_item_url_hash"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("source.id"))
    url: Mapped[str] = mapped_column(Text)
    url_hash: Mapped[str] = mapped_column(String(64), index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)
    outbound_links: Mapped[list | None] = mapped_column(JSON)
    # Lead image URL (spec D0). Populated either from the source feed's own
    # media:content/media:thumbnail/enclosure (feed.sources.base.RawItem.image_url,
    # set at collect time) or, failing that, the article page's
    # og:image/twitter:image meta tag (set at normalize time). NULL means
    # "no image found" -- a normal, common, and permanent outcome for many
    # items, not a pending state that gets filled in later. Additive column
    # (see feed/db.py's _ITEM_NEW_COLUMNS): existing rows read back NULL,
    # which is exactly the correct "no image" value here, so unlike the
    # story.status additive migration this needs no backfill -- there is no
    # wrong default to accidentally leave existing rows on.
    image_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)
    fetched_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_model_id: Mapped[str | None] = mapped_column(String(128))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("story.id"), index=True)
    stage: Mapped[Stage] = mapped_column(Enum(Stage), default=Stage.COLLECTED, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    story: Mapped["Story | None"] = relationship(back_populates="items")
    source: Mapped["Source"] = relationship()

class Story(Base):
    __tablename__ = "story"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    kind: Mapped[str | None] = mapped_column(String(32))
    first_seen: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    outlet_count: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float | None] = mapped_column(Float, index=True)
    score_breakdown: Mapped[dict | None] = mapped_column(JSON)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    # Phase 2 enrichment (spec 3.5, 3.2). `title` doubles as the Tier 1
    # canonical headline once enrichment has run -- before that it is
    # whatever the cluster stage seeded it with (the first item's title).
    summary: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
    analysis: Mapped[str | None] = mapped_column(Text)
    # "<provider name>:<model>", e.g. "claude-code:claude-code" or, on a
    # degraded Tier 2 fallback, "gemini:gemini-flash-latest" -- provenance
    # is a hard requirement (spec 3.5: "Every analysis records the provider
    # and model that produced it").
    analysis_provider: Mapped[str | None] = mapped_column(String(128))
    # Same provenance requirement as analysis_provider, but for the Tier 1
    # (BULK, once-per-story) headline/summary/category call -- spec: "which
    # provider AND model produced each summary/analysis". Set by
    # feed.stages.enrich.enrich_tier1 from RouteResult, e.g.
    # "groq:openai/gpt-oss-120b" or, on a Gemini 503 that failed over,
    # "mistral:mistral-medium-latest".
    summary_provider: Mapped[str | None] = mapped_column(String(128))
    analyzed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    status: Mapped[StoryStatus] = mapped_column(
        Enum(StoryStatus), default=StoryStatus.NEW, index=True
    )
    items: Mapped[list[Item]] = relationship(back_populates="story")

class Entity(Base):
    __tablename__ = "entity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    weight: Mapped[float] = mapped_column(Float, default=0.5)

class StoryEntity(Base):
    __tablename__ = "story_entity"
    story_id: Mapped[int] = mapped_column(ForeignKey("story.id"), primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id"), primary_key=True)


class ProviderStatus(Base):
    """Per-provider, per-UTC-day quota/health record, persisted so it
    survives restarts (requirement 2 of the multi-provider router task).
    One row per (provider, day); feed.providers.health.ProviderHealthTracker
    creates it lazily on first use each day. A brand-new table -- not new
    columns on an existing one -- so plain create_all() is enough to add it
    to an already-populated feed.db; no entry in feed.db._STORY_NEW_COLUMNS
    is needed.
    """
    __tablename__ = "provider_status"
    __table_args__ = (
        UniqueConstraint("provider", "day", name="uq_provider_status_provider_day"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    day: Mapped[str] = mapped_column(String(10))  # "YYYY-MM-DD", UTC
    requests: Mapped[int] = mapped_column(Integer, default=0)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    # Consecutive 429s only -- reset to 0 by any success OR any non-429
    # failure. This is what "repeated 429s ... skipped for the remainder
    # of the day" (requirement 2) is measured against, distinct from
    # `failures`, which counts every kind of failure.
    consecutive_429: Mapped[int] = mapped_column(Integer, default=0)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled_reason: Mapped[str | None] = mapped_column(String(256))
    last_error: Mapped[str | None] = mapped_column(Text)
    last_used_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
