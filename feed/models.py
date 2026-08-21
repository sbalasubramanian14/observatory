from __future__ import annotations
import enum
from datetime import datetime
from sqlalchemy import (
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, LargeBinary,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

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
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
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
