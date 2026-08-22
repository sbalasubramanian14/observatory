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
