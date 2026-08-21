# AI News Feed — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local pipeline that turns a broad crawl of AI news into deduplicated, importance-scored story clusters in SQLite, inspectable with a SQL client.

**Architecture:** A staged pipeline where SQLite is both the store and the work queue. Each `item` row carries a `stage` column; each stage claims rows for its stage, processes them, and advances them. Stages are independent and fail per-row, never per-batch. Sources, embedders, and scoring signals are plugin protocols registered by decorator.

**Tech Stack:** Python 3.14, SQLAlchemy 2.x, Pydantic 2.x, httpx, feedparser, trafilatura, numpy, fastembed (ONNX backend), sentence-transformers + torch (PyTorch backend, optional extra), pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-ai-news-feed-design.md`

## Global Constraints

- **Python 3.14 exactly.** Python 3.15 has no torch wheels. All venvs created with `py -3.14`.
- **No LLM calls anywhere in Phase 1.** Tiers 1 and 2 are Phase 2. Where the spec calls for LLM adjudication of ambiguous cluster pairs, Phase 1 installs a null adjudicator behind the same interface.
- **No network calls in tests.** Every source test uses a recorded fixture.
- **Every stored vector records its `embedding_model_id`.** Vectors from different models are not comparable (spec §3.3).
- **Full article text stays in local SQLite.** Nothing in Phase 1 publishes anything (spec §4.2).
- **Personal fit is not a pipeline signal.** The pipeline computes reader-independent importance only (spec §3.6).
- **Failure isolation is a requirement.** A stage catches per-row, records the error on that row, and continues (spec §3.1).
- Measured clustering constants from spec Appendix A: blend `0.6*cosine + 0.4*entities`, safe threshold band `0.46–0.56`, midpoint `0.50`.

---

## File Structure

```
pyproject.toml                     deps, Python pin, pytest config
feed/
  __init__.py
  config.py                        TOML -> typed Pydantic config
  db.py                            engine, session factory, create_all
  models.py                        ORM: Source, Item, Story, Entity, StoryEntity
  stages/
    base.py                        Stage protocol + claim/advance/error-isolation runner
    collect.py                     sources -> Item rows
    normalize.py                   canonical URL, text extraction, content-hash dedup
    embed.py                       batch embed, store vector + model id
    cluster.py                     candidate generation, blend, adjudication
    score.py                       four importance signals -> story.score
  sources/
    base.py                        RawItem dataclass + Source protocol
    registry.py                    @register decorator, lookup by id
    rss.py                         generic RSS/Atom source
    arxiv.py                       arXiv Atom API source
    hackernews.py                  HN Firebase API source
    github_releases.py             GitHub releases Atom source
  embedding/
    base.py                        Embedder protocol
    resolve.py                     device/backend "auto" resolution
    onnx_backend.py                fastembed
    torch_backend.py               sentence-transformers
  clustering/
    entities.py                    capitalised-token entity proxy
    signals.py                     cosine, entity jaccard, link overlap, time proximity
    adjudicate.py                  blend + decide + NullAdjudicator seam
  scoring/
    signals.py                     authority, velocity, novelty, entity weight
    combine.py                     weighted sum
  cli.py                           python -m feed
  __main__.py
tests/
  conftest.py                      in-memory db fixture, sample rows
  fixtures/                        recorded HTTP payloads
  golden/corpus.py                 22-item labelled clustering corpus
  test_*.py
```

---

### Task 1: Project scaffold and typed configuration

**Files:**
- Create: `pyproject.toml`, `feed/__init__.py`, `feed/config.py`, `feed.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_config(path: Path | None = None) -> Config`; `Config` with attributes `.database.url: str`, `.embedding.backend: str`, `.embedding.model: str`, `.embedding.device: str`, `.embedding.batch_size: int`, `.clustering.window_hours: int`, `.clustering.cosine_weight: float`, `.clustering.entity_weight: float`, `.clustering.merge_threshold: float`, `.scoring.weights: dict[str, float]`

- [ ] **Step 1: Create the venv and verify the Python pin**

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe --version   # must print 3.14.x
.venv/Scripts/python.exe -m pip install --upgrade pip
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "feed"
version = "0.1.0"
requires-python = ">=3.14,<3.15"
dependencies = [
    "sqlalchemy>=2.0",
    "pydantic>=2.0",
    "httpx>=0.27",
    "feedparser>=6.0",
    "trafilatura>=1.12",
    "numpy>=2.0",
    "fastembed>=0.8",
]

[project.optional-dependencies]
torch = ["sentence-transformers>=6.0", "torch>=2.9"]
dev = ["pytest>=8.0", "pytest-cov"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Install and confirm every dependency resolves on 3.14**

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -c "import sqlalchemy, pydantic, httpx, feedparser, trafilatura, numpy, fastembed; print('all imports ok')"
```

Expected: `all imports ok`. If `trafilatura` fails to build on 3.14, stop and report — do not substitute a different extractor without raising it, because extraction quality directly sets the ceiling on clustering quality.

- [ ] **Step 4: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from feed.config import load_config

def test_loads_defaults_when_file_absent(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.embedding.device == "auto"
    assert cfg.embedding.batch_size == 256
    # merge_threshold is an empirically derived constant that Task 12 may
    # revise from the golden-set band. Assert it is sane, not exact, so the
    # two tasks are not coupled.
    assert 0.30 <= cfg.clustering.merge_threshold <= 0.70

def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[embedding]\ndevice = \"cpu\"\nbatch_size = 64\n"
        "[clustering]\nmerge_threshold = 0.62\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.embedding.device == "cpu"
    assert cfg.embedding.batch_size == 64
    assert cfg.clustering.merge_threshold == 0.62
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"  # untouched default

def test_rejects_unknown_device(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text("[embedding]\ndevice = \"tpu\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)

def test_clustering_weights_must_sum_to_one(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[clustering]\ncosine_weight = 0.9\nentity_weight = 0.4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_config(p)
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.config'`

- [ ] **Step 6: Implement `feed/config.py`**

```python
# feed/config.py
from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///feed.db"

class EmbeddingConfig(BaseModel):
    backend: Literal["auto", "torch", "onnx"] = "auto"
    model: str = "BAAI/bge-small-en-v1.5"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: int = Field(default=256, gt=0, le=1024)

class ClusteringConfig(BaseModel):
    window_hours: int = Field(default=48, gt=0)
    cosine_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    entity_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    merge_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ClusteringConfig":
        total = self.cosine_weight + self.entity_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"cosine_weight + entity_weight must sum to 1.0, got {total}"
            )
        return self

class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "authority": 0.25,
            "velocity": 0.40,
            "novelty": 0.20,
            "entity": 0.15,
        }
    )

class Config(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path is not None else Path("feed.toml")
    if not path.exists():
        return Config()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return Config.model_validate(raw)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 8: Write the default `feed.toml` and commit**

```toml
# feed.toml
[database]
url = "sqlite:///feed.db"

[embedding]
backend = "auto"
model = "BAAI/bge-small-en-v1.5"
device = "auto"
batch_size = 256

[clustering]
window_hours = 48
cosine_weight = 0.6
entity_weight = 0.4
merge_threshold = 0.50
```

```bash
git add pyproject.toml feed/ feed.toml tests/test_config.py
git commit -m "feat: project scaffold and typed configuration"
```

---

### Task 2: Database schema and session

**Files:**
- Create: `feed/models.py`, `feed/db.py`
- Test: `tests/conftest.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `Config` from Task 1
- Produces: ORM classes `Source`, `Item`, `Story`, `Entity`, `StoryEntity`; `Stage` enum with members `COLLECTED`, `NORMALIZED`, `EMBEDDED`, `CLUSTERED`, `SCORED`, `FAILED`; `make_engine(url: str)`, `make_session_factory(engine)`, `create_all(engine)`

- [ ] **Step 1: Write the failing test**

```python
# tests/conftest.py
import pytest
from sqlalchemy.orm import Session
from feed.db import make_engine, make_session_factory, create_all

@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite://")   # in-memory
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s
```

```python
# tests/test_models.py
from datetime import datetime, timezone
from feed.models import Source, Item, Story, Stage

def test_item_starts_in_collected_stage(session):
    src = Source(id="rss:example", plugin="rss", config={"url": "http://x"}, cadence_minutes=30)
    session.add(src)
    item = Item(source_id="rss:example", url="http://x/1", url_hash="h1", title="T")
    session.add(item)
    session.commit()
    assert item.stage is Stage.COLLECTED
    assert item.error is None

def test_url_hash_is_unique(session):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.add(Item(source_id="s", url="http://a", url_hash="dup", title="A"))
    session.commit()
    session.add(Item(source_id="s", url="http://b", url_hash="dup", title="B"))
    import pytest, sqlalchemy.exc
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.commit()

def test_story_tracks_item_count_and_updated_at(session):
    st = Story(title="S", first_seen=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc), item_count=0)
    session.add(st)
    session.commit()
    assert st.id is not None
    assert st.score is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.models'`

- [ ] **Step 3: Implement `feed/models.py`**

```python
# feed/models.py
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
```

- [ ] **Step 4: Implement `feed/db.py`**

```python
# feed/db.py
from __future__ import annotations
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from feed.models import Base

def make_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine

def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)

def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add feed/models.py feed/db.py tests/conftest.py tests/test_models.py
git commit -m "feat: database schema and session factory"
```

---

### Task 3: Stage runner with per-row failure isolation

**Files:**
- Create: `feed/stages/__init__.py`, `feed/stages/base.py`
- Test: `tests/test_stage_runner.py`

**Interfaces:**
- Consumes: `Item`, `Stage`, session factory from Task 2
- Produces: `run_stage(session, *, name: str, claim_stage: Stage, next_stage: Stage, handler: Callable[[Session, Item], None], limit: int = 100) -> StageResult`; `StageResult` with `.processed: int`, `.failed: int`, `.errors: list[tuple[int, str]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage_runner.py
from feed.models import Item, Source, Stage
from feed.stages.base import run_stage

def _seed(session, n=3):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    for i in range(n):
        session.add(Item(source_id="s", url=f"http://x/{i}", url_hash=f"h{i}", title=f"T{i}"))
    session.commit()

def test_advances_rows_to_next_stage(session):
    _seed(session)
    res = run_stage(session, name="noop", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None)
    assert res.processed == 3 and res.failed == 0
    assert all(i.stage is Stage.NORMALIZED for i in session.query(Item).all())

def test_one_bad_row_does_not_stop_the_batch(session):
    _seed(session)
    def handler(s, item):
        if item.url_hash == "h1":
            raise RuntimeError("boom")
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=handler)
    assert res.processed == 2 and res.failed == 1
    by_hash = {i.url_hash: i for i in session.query(Item).all()}
    assert by_hash["h0"].stage is Stage.NORMALIZED
    assert by_hash["h2"].stage is Stage.NORMALIZED
    assert by_hash["h1"].stage is Stage.FAILED
    assert "boom" in by_hash["h1"].error

def test_only_claims_matching_stage(session):
    _seed(session)
    session.query(Item).filter_by(url_hash="h0").one().stage = Stage.EMBEDDED
    session.commit()
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None)
    assert res.processed == 2

def test_limit_is_respected(session):
    _seed(session, n=10)
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None, limit=4)
    assert res.processed == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stage_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.stages'`

- [ ] **Step 3: Implement `feed/stages/base.py`**

```python
# feed/stages/base.py
from __future__ import annotations
import logging
import traceback
from dataclasses import dataclass, field
from typing import Callable
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import Item, Stage

log = logging.getLogger(__name__)

@dataclass
class StageResult:
    name: str
    processed: int = 0
    failed: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)

Handler = Callable[[Session, Item], None]

def run_stage(
    session: Session,
    *,
    name: str,
    claim_stage: Stage,
    next_stage: Stage,
    handler: Handler,
    limit: int = 100,
) -> StageResult:
    """Claim items at `claim_stage`, run `handler` on each, advance to `next_stage`.

    Failure is isolated per row: a raising handler marks that row FAILED with
    the traceback and the batch continues. One broken source must never stall
    the pipeline.
    """
    result = StageResult(name=name)
    stmt = select(Item).where(Item.stage == claim_stage).order_by(Item.id).limit(limit)
    items = list(session.scalars(stmt))

    for item in items:
        try:
            handler(session, item)
        except Exception as exc:
            session.rollback()
            fresh = session.get(Item, item.id)
            if fresh is not None:
                fresh.stage = Stage.FAILED
                fresh.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                session.commit()
            result.failed += 1
            result.errors.append((item.id, str(exc)))
            log.warning("stage=%s item=%s failed: %s", name, item.id, exc)
        else:
            item.stage = next_stage
            item.error = None
            session.commit()
            result.processed += 1

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stage_runner.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add feed/stages/ tests/test_stage_runner.py
git commit -m "feat: stage runner with per-row failure isolation"
```

---

### Task 4: Source protocol, registry, and the generic RSS source

**Files:**
- Create: `feed/sources/__init__.py`, `feed/sources/base.py`, `feed/sources/registry.py`, `feed/sources/rss.py`
- Test: `tests/fixtures/sample_rss.xml`, `tests/test_sources_rss.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `RawItem` dataclass with fields `url: str`, `title: str`, `summary: str | None`, `published_at: datetime | None`, `outbound_links: list[str]`; `Source` protocol with `id: str`, `fetch(since: datetime | None) -> Iterable[RawItem]`; `@register(plugin_name)` decorator; `build_source(plugin: str, source_id: str, config: dict) -> Source`; `canonical_url(url: str) -> str`; `url_hash(url: str) -> str`

- [ ] **Step 1: Write the failing test**

Save a trimmed real feed as `tests/fixtures/sample_rss.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example AI Blog</title>
  <item>
    <title>DeepSeek releases V4</title>
    <link>https://example.com/deepseek-v4?utm_source=rss&amp;utm_medium=feed</link>
    <description>The lab published weights under an MIT license.</description>
    <pubDate>Tue, 18 Aug 2026 09:00:00 GMT</pubDate>
  </item>
  <item>
    <title>EU delays AI Act</title>
    <link>https://example.com/eu-delay</link>
    <description>Enforcement postponed by eighteen months.</description>
    <pubDate>Tue, 18 Aug 2026 11:30:00 GMT</pubDate>
  </item>
</channel></rss>
```

```python
# tests/test_sources_rss.py
from datetime import datetime, timezone
from pathlib import Path
from feed.sources.base import canonical_url, url_hash
from feed.sources.registry import build_source

FIXTURE = Path(__file__).parent / "fixtures" / "sample_rss.xml"

def test_canonical_url_strips_tracking_params():
    got = canonical_url("https://example.com/a?utm_source=rss&utm_medium=feed&id=7")
    assert got == "https://example.com/a?id=7"

def test_canonical_url_is_stable_across_trivial_differences():
    a = canonical_url("https://Example.com/a/")
    b = canonical_url("http://example.com/a")
    assert a == b
    assert url_hash(a) == url_hash(b)

def test_rss_source_parses_fixture():
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    items = list(src.fetch(since=None))
    assert len(items) == 2
    first = items[0]
    assert first.title == "DeepSeek releases V4"
    assert first.url == "https://example.com/deepseek-v4"
    assert first.published_at == datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

def test_rss_source_filters_by_since():
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    cutoff = datetime(2026, 8, 18, 10, 0, tzinfo=timezone.utc)
    items = list(src.fetch(since=cutoff))
    assert [i.title for i in items] == ["EU delays AI Act"]

def test_unknown_plugin_raises():
    import pytest
    with pytest.raises(KeyError):
        build_source("nope", "x", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources_rss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.sources'`

- [ ] **Step 3: Implement `feed/sources/base.py`**

```python
# feed/sources/base.py
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_src")

@dataclass(slots=True)
class RawItem:
    url: str
    title: str
    summary: str | None = None
    published_at: datetime | None = None
    outbound_links: list[str] = field(default_factory=list)

class Source(Protocol):
    id: str
    def fetch(self, since: datetime | None) -> Iterable[RawItem]: ...

def canonical_url(url: str) -> str:
    """Normalise a URL so the same article from two places hashes identically."""
    parts = urlsplit(url.strip())
    scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not any(k.lower().startswith(p) for p in TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path, query, ""))

def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Implement `feed/sources/registry.py`**

```python
# feed/sources/registry.py
from __future__ import annotations
from typing import Callable, TypeVar
from feed.sources.base import Source

_REGISTRY: dict[str, Callable[..., Source]] = {}
T = TypeVar("T")

def register(plugin: str) -> Callable[[T], T]:
    def wrap(cls: T) -> T:
        if plugin in _REGISTRY:
            raise ValueError(f"source plugin already registered: {plugin}")
        _REGISTRY[plugin] = cls  # type: ignore[assignment]
        return cls
    return wrap

def build_source(plugin: str, source_id: str, config: dict) -> Source:
    if plugin not in _REGISTRY:
        raise KeyError(f"unknown source plugin: {plugin!r}. known: {sorted(_REGISTRY)}")
    return _REGISTRY[plugin](source_id=source_id, **config)

def known_plugins() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 5: Implement `feed/sources/rss.py` and the package import**

```python
# feed/sources/rss.py
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

@register("rss")
class RssSource:
    """Generic RSS/Atom source.

    `url` fetches over HTTP; `path` reads a local file and exists so tests
    never touch the network.
    """

    def __init__(self, source_id: str, url: str | None = None, path: str | None = None,
                 timeout: float = 20.0):
        if not url and not path:
            raise ValueError("rss source needs either url or path")
        self.id = source_id
        self.url = url
        self.path = path
        self.timeout = timeout

    def _raw(self) -> str | bytes:
        if self.path:
            return Path(self.path).read_bytes()
        resp = httpx.get(self.url, timeout=self.timeout,
                         headers={"User-Agent": "feed/0.1 (personal reader)"},
                         follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        parsed = feedparser.parse(self._raw())
        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue
            published = None
            tm = entry.get("published_parsed") or entry.get("updated_parsed")
            if tm:
                published = datetime(*tm[:6], tzinfo=timezone.utc)
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(link),
                title=(entry.get("title") or "").strip(),
                summary=(entry.get("summary") or None),
                published_at=published,
            )
```

```python
# feed/sources/__init__.py
from feed.sources import rss  # noqa: F401  (registers the plugin)
from feed.sources.registry import build_source, known_plugins, register  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources_rss.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add feed/sources/ tests/fixtures/sample_rss.xml tests/test_sources_rss.py
git commit -m "feat: source protocol, plugin registry, and RSS source"
```

---

### Task 5: arXiv, Hacker News, and GitHub Releases sources

**Files:**
- Create: `feed/sources/arxiv.py`, `feed/sources/hackernews.py`, `feed/sources/github_releases.py`
- Modify: `feed/sources/__init__.py`
- Test: `tests/fixtures/sample_arxiv.xml`, `tests/fixtures/sample_hn.json`, `tests/test_sources_more.py`

**Interfaces:**
- Consumes: `RawItem`, `register`, `canonical_url` from Task 4
- Produces: plugins registered under names `arxiv`, `hackernews`, `github_releases`

- [ ] **Step 1: Save fixtures**

`tests/fixtures/sample_arxiv.xml` — an arXiv Atom response trimmed to two entries:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2607.09510v1</id>
    <updated>2026-07-13T10:04:00Z</updated>
    <published>2026-07-13T10:04:00Z</published>
    <title>Failure as a Process: An Anatomy of CLI Coding Agent Trajectories</title>
    <summary>We analyse 1184 failed trajectories from three coding agent scaffolds.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2606.17799v2</id>
    <updated>2026-06-20T08:00:00Z</updated>
    <published>2026-06-19T08:00:00Z</published>
    <title>Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering</title>
    <summary>Benchmarks measure one-shot generation while real work is iterative.</summary>
  </entry>
</feed>
```

`tests/fixtures/sample_hn.json`:

```json
[
  {"id": 1, "title": "Show HN: DeepSeek V4 weights are up", "url": "https://example.com/v4", "score": 412, "time": 1787000000, "type": "story"},
  {"id": 2, "title": "Low score story", "url": "https://example.com/meh", "score": 12, "time": 1787000100, "type": "story"},
  {"id": 3, "title": "Ask HN: no url", "score": 300, "time": 1787000200, "type": "story"}
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_sources_more.py
import json
from datetime import datetime, timezone
from pathlib import Path
from feed.sources.registry import build_source

FIX = Path(__file__).parent / "fixtures"

def test_arxiv_builds_abs_urls_and_titles():
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(FIX / "sample_arxiv.xml")})
    items = list(src.fetch(since=None))
    assert len(items) == 2
    assert items[0].url == "https://arxiv.org/abs/2607.09510"
    assert "Failure as a Process" in items[0].title
    assert items[0].published_at == datetime(2026, 7, 13, 10, 4, tzinfo=timezone.utc)

def test_arxiv_strips_version_suffix_so_v1_and_v2_share_a_url():
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(FIX / "sample_arxiv.xml")})
    urls = [i.url for i in src.fetch(since=None)]
    assert all("v1" not in u and "v2" not in u for u in urls)

def test_hackernews_applies_min_score_and_skips_urlless(tmp_path):
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    items = list(src.fetch(since=None))
    assert [i.title for i in items] == ["Show HN: DeepSeek V4 weights are up"]

def test_github_releases_url_is_built_from_repo():
    src = build_source("github_releases", "gh:vllm", {"repo": "vllm-project/vllm"})
    assert src.feed_url == "https://github.com/vllm-project/vllm/releases.atom"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources_more.py -v`
Expected: FAIL — `KeyError: unknown source plugin: 'arxiv'`

- [ ] **Step 4: Implement `feed/sources/arxiv.py`**

```python
# feed/sources/arxiv.py
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem
from feed.sources.registry import register

_ABS = re.compile(r"arxiv\.org/abs/(?P<id>[\d.]+?)(?:v\d+)?$")

@register("arxiv")
class ArxivSource:
    """arXiv Atom API. Version suffixes are stripped so v1 and v2 of the same
    paper collapse to one URL and therefore one item."""

    API = "https://export.arxiv.org/api/query"

    def __init__(self, source_id: str, categories: list[str] | None = None,
                 max_results: int = 100, path: str | None = None, timeout: float = 30.0):
        self.id = source_id
        self.categories = categories or ["cs.AI", "cs.CL", "cs.LG"]
        self.max_results = max_results
        self.path = path
        self.timeout = timeout

    def _raw(self) -> str | bytes:
        if self.path:
            return Path(self.path).read_bytes()
        query = "+OR+".join(f"cat:{c}" for c in self.categories)
        url = (f"{self.API}?search_query={query}&sortBy=submittedDate"
               f"&sortOrder=descending&max_results={self.max_results}")
        resp = httpx.get(url, timeout=self.timeout,
                         headers={"User-Agent": "feed/0.1 (personal reader)"})
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        parsed = feedparser.parse(self._raw())
        for entry in parsed.entries:
            match = _ABS.search(entry.get("id", ""))
            if not match:
                continue
            url = f"https://arxiv.org/abs/{match.group('id')}"
            tm = entry.get("published_parsed") or entry.get("updated_parsed")
            published = datetime(*tm[:6], tzinfo=timezone.utc) if tm else None
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=url,
                title=" ".join((entry.get("title") or "").split()),
                summary=" ".join((entry.get("summary") or "").split()) or None,
                published_at=published,
            )
```

- [ ] **Step 5: Implement `feed/sources/hackernews.py` and `feed/sources/github_releases.py`**

```python
# feed/sources/hackernews.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

@register("hackernews")
class HackerNewsSource:
    """Hacker News top stories above a score floor.

    The score floor is the point: HN volume is enormous and low-score stories
    are noise the pipeline should never pay to embed.
    """

    TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"

    def __init__(self, source_id: str, min_score: int = 100, limit: int = 100,
                 path: str | None = None, timeout: float = 20.0):
        self.id = source_id
        self.min_score = min_score
        self.limit = limit
        self.path = path
        self.timeout = timeout

    def _stories(self) -> list[dict]:
        if self.path:
            return json.loads(Path(self.path).read_text(encoding="utf-8"))
        with httpx.Client(timeout=self.timeout) as client:
            ids = client.get(self.TOP).json()[: self.limit]
            return [client.get(self.ITEM.format(id=i)).json() for i in ids]

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        for story in self._stories():
            if not story or story.get("type") != "story":
                continue
            if not story.get("url"):
                continue
            if story.get("score", 0) < self.min_score:
                continue
            published = datetime.fromtimestamp(story["time"], tz=timezone.utc)
            if since is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(story["url"]),
                title=story.get("title", "").strip(),
                summary=None,
                published_at=published,
            )
```

```python
# feed/sources/github_releases.py
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

@register("github_releases")
class GithubReleasesSource:
    """Release notes for a repo via its public releases.atom feed."""

    def __init__(self, source_id: str, repo: str | None = None,
                 path: str | None = None, timeout: float = 20.0):
        if not repo and not path:
            raise ValueError("github_releases needs repo or path")
        self.id = source_id
        self.repo = repo
        self.path = path
        self.timeout = timeout

    @property
    def feed_url(self) -> str:
        return f"https://github.com/{self.repo}/releases.atom"

    def _raw(self) -> str | bytes:
        if self.path:
            return Path(self.path).read_bytes()
        resp = httpx.get(self.feed_url, timeout=self.timeout,
                         headers={"User-Agent": "feed/0.1 (personal reader)"},
                         follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        parsed = feedparser.parse(self._raw())
        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue
            tm = entry.get("updated_parsed") or entry.get("published_parsed")
            published = datetime(*tm[:6], tzinfo=timezone.utc) if tm else None
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(link),
                title=f"{self.repo}: {(entry.get('title') or '').strip()}",
                summary=(entry.get("summary") or None),
                published_at=published,
            )
```

```python
# feed/sources/__init__.py
from feed.sources import arxiv, github_releases, hackernews, rss  # noqa: F401
from feed.sources.registry import build_source, known_plugins, register  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources_more.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add feed/sources/ tests/fixtures/ tests/test_sources_more.py
git commit -m "feat: arXiv, Hacker News, and GitHub Releases sources"
```

---

### Task 6: Collect stage

**Files:**
- Create: `feed/stages/collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `build_source` (Task 4/5), `Source`/`Item` models (Task 2), `url_hash` (Task 4)
- Produces: `collect(session, *, now: datetime | None = None) -> CollectResult` with `.new_items: int`, `.skipped_duplicates: int`, `.source_errors: dict[str, str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect.py
from datetime import datetime, timedelta, timezone
from pathlib import Path
from feed.models import Item, Source, Stage
from feed.stages.collect import collect

FIX = Path(__file__).parent / "fixtures" / "sample_rss.xml"
NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)

def _add_source(session, **kw):
    defaults = dict(id="rss:example", plugin="rss", config={"path": str(FIX)},
                    cadence_minutes=30, enabled=True)
    defaults.update(kw)
    session.add(Source(**defaults))
    session.commit()

def test_collect_inserts_items_at_collected_stage(session):
    _add_source(session)
    res = collect(session, now=NOW)
    assert res.new_items == 2
    items = session.query(Item).all()
    assert len(items) == 2
    assert all(i.stage is Stage.COLLECTED for i in items)

def test_collect_is_idempotent_on_url_hash(session):
    _add_source(session)
    collect(session, now=NOW)
    res = collect(session, now=NOW + timedelta(hours=1))
    assert res.new_items == 0
    assert res.skipped_duplicates == 2
    assert session.query(Item).count() == 2

def test_disabled_source_is_skipped(session):
    _add_source(session, enabled=False)
    res = collect(session, now=NOW)
    assert res.new_items == 0

def test_cadence_prevents_early_refetch(session):
    _add_source(session, cadence_minutes=60)
    collect(session, now=NOW)
    session.query(Item).delete()
    session.commit()
    res = collect(session, now=NOW + timedelta(minutes=10))
    assert res.new_items == 0          # too soon
    res = collect(session, now=NOW + timedelta(minutes=61))
    assert res.new_items == 2

def test_broken_source_is_recorded_and_does_not_raise(session):
    _add_source(session, id="bad", config={"path": "/nonexistent.xml"})
    res = collect(session, now=NOW)
    assert "bad" in res.source_errors
    src = session.get(Source, "bad")
    assert src.consecutive_failures == 1
    assert src.last_error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.stages.collect'`

- [ ] **Step 3: Implement `feed/stages/collect.py`**

```python
# feed/stages/collect.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
import feed.sources  # noqa: F401  (registers plugins)
from feed.models import Item, Source, Stage
from feed.sources.base import url_hash
from feed.sources.registry import build_source

log = logging.getLogger(__name__)

@dataclass
class CollectResult:
    new_items: int = 0
    skipped_duplicates: int = 0
    source_errors: dict[str, str] = field(default_factory=dict)

def collect(session: Session, *, now: datetime | None = None) -> CollectResult:
    now = now or datetime.now(timezone.utc)
    result = CollectResult()

    for src in session.scalars(select(Source).where(Source.enabled.is_(True))):
        if src.last_run_at is not None:
            due = src.last_run_at + timedelta(minutes=src.cadence_minutes)
            if now < due:
                continue
        try:
            plugin = build_source(src.plugin, src.id, dict(src.config or {}))
            raw_items = list(plugin.fetch(since=src.last_run_at))
        except Exception as exc:
            src.consecutive_failures += 1
            src.last_error = f"{type(exc).__name__}: {exc}"
            session.commit()
            result.source_errors[src.id] = src.last_error
            log.warning("source=%s fetch failed: %s", src.id, exc)
            continue

        for raw in raw_items:
            h = url_hash(raw.url)
            exists = session.scalar(select(Item.id).where(Item.url_hash == h))
            if exists:
                result.skipped_duplicates += 1
                continue
            session.add(Item(
                source_id=src.id,
                url=raw.url,
                url_hash=h,
                title=raw.title,
                summary=raw.summary,
                outbound_links=raw.outbound_links or [],
                published_at=raw.published_at,
                fetched_at=now,
                stage=Stage.COLLECTED,
            ))
            result.new_items += 1

        src.last_run_at = now
        src.last_error = None
        src.consecutive_failures = 0
        session.commit()

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_collect.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add feed/stages/collect.py tests/test_collect.py
git commit -m "feat: collect stage with cadence, dedup, and source health"
```

---

### Task 7: Normalize stage

**Files:**
- Create: `feed/stages/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Consumes: `run_stage` (Task 3), `Item`/`Stage` (Task 2)
- Produces: `content_hash(text: str) -> str`; `normalize_item(session, item) -> None`; `normalize(session, limit=100) -> StageResult`

Note: extraction is skipped when the source already supplied a usable summary and the item is an arXiv abstract — refetching arXiv HTML adds nothing. This keeps the stage cheap and polite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
import pytest
from feed.models import Item, Source, Stage
from feed.stages.normalize import content_hash, normalize

def _seed(session, **kw):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    defaults = dict(source_id="s", url="https://arxiv.org/abs/2607.09510",
                    url_hash="h1", title="A paper", summary="An abstract with real words.")
    defaults.update(kw)
    item = Item(**defaults)
    session.add(item)
    session.commit()
    return item

def test_content_hash_ignores_whitespace_differences():
    assert content_hash("hello   world\n") == content_hash("hello world")

def test_content_hash_differs_for_different_text():
    assert content_hash("a") != content_hash("b")

def test_normalize_populates_text_and_hash_from_summary(session):
    item = _seed(session)
    res = normalize(session)
    assert res.processed == 1
    session.refresh(item)
    assert item.stage is Stage.NORMALIZED
    assert item.text == "An abstract with real words."
    assert item.content_hash is not None

def test_near_duplicate_content_is_marked_failed_not_advanced(session):
    _seed(session)
    normalize(session)
    dup = _seed(session, url_hash="h2", url="https://other.example/x",
                summary="An abstract with real words.")
    res = normalize(session)
    session.refresh(dup)
    assert res.processed == 0 and res.failed == 1
    assert dup.stage is Stage.FAILED
    assert "duplicate" in dup.error.lower()

def test_item_with_no_usable_text_fails_cleanly(session):
    item = _seed(session, url_hash="h3", summary="", title="")
    res = normalize(session)
    session.refresh(item)
    assert item.stage is Stage.FAILED
    assert "no text" in item.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.stages.normalize'`

- [ ] **Step 3: Implement `feed/stages/normalize.py`**

```python
# feed/stages/normalize.py
from __future__ import annotations
import hashlib
import re
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import Item, Stage
from feed.stages.base import StageResult, run_stage

_WS = re.compile(r"\s+")
MIN_TEXT_CHARS = 20

class DuplicateContent(Exception):
    pass

def content_hash(text: str) -> str:
    collapsed = _WS.sub(" ", text or "").strip().lower()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()

def _extract(item: Item) -> str:
    """Prefer the summary the source gave us; fall back to fetching the page.

    arXiv abstracts and most RSS descriptions are already the best available
    text, and refetching adds latency, failure modes, and load on the source
    for no gain.
    """
    if item.summary and len(item.summary.strip()) >= MIN_TEXT_CHARS:
        return _WS.sub(" ", item.summary).strip()
    if item.url.startswith("https://arxiv.org/abs/"):
        return ""
    import trafilatura
    downloaded = trafilatura.fetch_url(item.url)
    if not downloaded:
        return ""
    return _WS.sub(" ", trafilatura.extract(downloaded) or "").strip()

def normalize_item(session: Session, item: Item) -> None:
    text = _extract(item)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(f"no text extracted for {item.url}")
    digest = content_hash(text)
    clash = session.scalar(
        select(Item.id).where(Item.content_hash == digest, Item.id != item.id)
    )
    if clash:
        raise DuplicateContent(f"duplicate content of item {clash}")
    item.text = text
    item.content_hash = digest

def normalize(session: Session, limit: int = 100) -> StageResult:
    return run_stage(
        session, name="normalize", claim_stage=Stage.COLLECTED,
        next_stage=Stage.NORMALIZED, handler=normalize_item, limit=limit,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_normalize.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add feed/stages/normalize.py tests/test_normalize.py
git commit -m "feat: normalize stage with content-hash dedup"
```

---

### Task 8: Embedding backends and device resolution

**Files:**
- Create: `feed/embedding/__init__.py`, `feed/embedding/base.py`, `feed/embedding/resolve.py`, `feed/embedding/onnx_backend.py`, `feed/embedding/torch_backend.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `EmbeddingConfig` (Task 1)
- Produces: `Embedder` protocol with `model_id: str`, `dimensions: int`, `encode(texts: list[str]) -> np.ndarray`; `resolve(cfg: EmbeddingConfig) -> tuple[str, str, str]` returning `(backend, model, device)`; `build_embedder(cfg: EmbeddingConfig) -> Embedder`; `pack(vec: np.ndarray) -> bytes`; `unpack(blob: bytes) -> np.ndarray`

Spec §3.3: `auto` resolves to torch+CUDA when a GPU is present, otherwise onnx+MiniLM on CPU, which was the fastest measured CPU configuration.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
import numpy as np
import pytest
from feed.config import EmbeddingConfig
from feed.embedding.base import pack, unpack
from feed.embedding.resolve import resolve
from feed.embedding import build_embedder

def test_pack_roundtrips_exactly():
    v = np.random.rand(384).astype(np.float32)
    assert np.array_equal(unpack(pack(v)), v)

def test_explicit_settings_are_passed_through():
    cfg = EmbeddingConfig(backend="torch", model="BAAI/bge-small-en-v1.5", device="cpu")
    assert resolve(cfg) == ("torch", "BAAI/bge-small-en-v1.5", "cpu")

def test_auto_without_gpu_picks_onnx_minilm_cpu(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    cfg = EmbeddingConfig(backend="auto", device="auto")
    assert resolve(cfg) == ("onnx", "sentence-transformers/all-MiniLM-L6-v2", "cpu")

def test_auto_with_gpu_picks_torch_bge_cuda(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: True)
    cfg = EmbeddingConfig(backend="auto", device="auto")
    assert resolve(cfg) == ("torch", "BAAI/bge-small-en-v1.5", "cuda")

def test_explicit_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda requested"):
        resolve(EmbeddingConfig(device="cuda", backend="torch"))

@pytest.mark.slow
def test_onnx_embedder_produces_normalisable_vectors():
    cfg = EmbeddingConfig(backend="onnx",
                          model="sentence-transformers/all-MiniLM-L6-v2",
                          device="cpu", batch_size=8)
    emb = build_embedder(cfg)
    V = emb.encode(["DeepSeek releases V4", "EU delays the AI Act"])
    assert V.shape == (2, emb.dimensions)
    assert emb.model_id.endswith("all-MiniLM-L6-v2")
    sim = float(V[0] @ V[1] / (np.linalg.norm(V[0]) * np.linalg.norm(V[1])))
    assert -1.0 <= sim <= 1.0
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["slow: downloads a model or takes more than a few seconds"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embedding.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.embedding'`

- [ ] **Step 3: Implement `feed/embedding/base.py` and `feed/embedding/resolve.py`**

```python
# feed/embedding/base.py
from __future__ import annotations
from typing import Protocol
import numpy as np

class Embedder(Protocol):
    model_id: str
    dimensions: int
    def encode(self, texts: list[str]) -> np.ndarray: ...

def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()

def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
```

```python
# feed/embedding/resolve.py
from __future__ import annotations
from feed.config import EmbeddingConfig

CPU_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GPU_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())

def resolve(cfg: EmbeddingConfig) -> tuple[str, str, str]:
    """Return (backend, model, device).

    Measured rationale (spec Appendix A): on CPU, ONNX + MiniLM is the fastest
    configuration at ~90 docs/s. With a GPU, torch + bge-small reaches ~203
    docs/s while also giving a 512-token window instead of 256, so the GPU
    buys the better model rather than only speed.
    """
    has_gpu = cuda_available()

    device = cfg.device
    if device == "auto":
        device = "cuda" if has_gpu else "cpu"
    elif device == "cuda" and not has_gpu:
        raise RuntimeError("cuda requested but no CUDA device is available")

    backend = cfg.backend
    if backend == "auto":
        backend = "torch" if device == "cuda" else "onnx"
    if backend == "onnx" and device == "cuda":
        raise RuntimeError("onnx backend is CPU-only in this project; use backend=torch")

    model = cfg.model
    if model == EmbeddingConfig().model and backend == "onnx":
        # the bge ONNX export is anomalously slow (spec Appendix A); prefer MiniLM
        model = CPU_DEFAULT_MODEL
    return backend, model, device
```

- [ ] **Step 4: Implement both backends and the factory**

```python
# feed/embedding/onnx_backend.py
from __future__ import annotations
import numpy as np

class OnnxEmbedder:
    def __init__(self, model: str, batch_size: int = 256):
        from fastembed import TextEmbedding
        self.model_id = model
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model)
        self.dimensions = int(next(iter(self._model.embed(["probe"]))).shape[0])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.asarray(
            list(self._model.embed(texts, batch_size=self.batch_size)), dtype=np.float32
        )
```

```python
# feed/embedding/torch_backend.py
from __future__ import annotations
import numpy as np

class TorchEmbedder:
    def __init__(self, model: str, device: str = "cpu", batch_size: int = 256):
        from sentence_transformers import SentenceTransformer
        self.model_id = model
        self.batch_size = batch_size
        self._model = SentenceTransformer(model, device=device)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size,
                               show_progress_bar=False),
            dtype=np.float32,
        )
```

```python
# feed/embedding/__init__.py
from __future__ import annotations
from feed.config import EmbeddingConfig
from feed.embedding.base import Embedder, pack, unpack  # noqa: F401
from feed.embedding.resolve import resolve

def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    backend, model, device = resolve(cfg)
    if backend == "onnx":
        from feed.embedding.onnx_backend import OnnxEmbedder
        return OnnxEmbedder(model, batch_size=cfg.batch_size)
    from feed.embedding.torch_backend import TorchEmbedder
    return TorchEmbedder(model, device=device, batch_size=cfg.batch_size)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embedding.py -v`
Expected: 6 passed (the slow one downloads a ~90 MB model on first run)

- [ ] **Step 6: Commit**

```bash
git add feed/embedding/ tests/test_embedding.py pyproject.toml
git commit -m "feat: torch and onnx embedding backends with auto device resolution"
```

---

### Task 9: Embed stage

**Files:**
- Create: `feed/stages/embed.py`
- Test: `tests/test_embed_stage.py`

**Interfaces:**
- Consumes: `Embedder`, `pack` (Task 8), `Item`/`Stage` (Task 2)
- Produces: `embed(session, embedder, limit=256) -> StageResult`

Batching matters here: the stage runner processes one row at a time, but embedding is 30x faster in batches, so this stage deliberately does its own batched claim rather than using `run_stage`. Batch size must be pinned — an unpinned batch peaked at 5.6 GB RSS in the spike (spec Appendix A).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_stage.py
import numpy as np
from feed.embedding.base import unpack
from feed.models import Item, Source, Stage
from feed.stages.embed import embed

class FakeEmbedder:
    model_id = "fake/model-v1"
    dimensions = 4
    def __init__(self): self.calls = []
    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.arange(4, dtype=np.float32), (len(texts), 1))

class ExplodingEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("gpu on fire")

def _seed(session, n=3):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    for i in range(n):
        session.add(Item(source_id="s", url=f"http://x/{i}", url_hash=f"h{i}",
                         title=f"T{i}", text=f"body {i}", stage=Stage.NORMALIZED))
    session.commit()

def test_embeds_and_advances(session):
    _seed(session)
    emb = FakeEmbedder()
    res = embed(session, emb)
    assert res.processed == 3
    items = session.query(Item).all()
    assert all(i.stage is Stage.EMBEDDED for i in items)
    assert all(i.embedding_model_id == "fake/model-v1" for i in items)
    assert np.array_equal(unpack(items[0].embedding), np.arange(4, dtype=np.float32))

def test_encodes_in_one_batched_call(session):
    _seed(session, n=5)
    emb = FakeEmbedder()
    embed(session, emb, limit=5)
    assert len(emb.calls) == 1 and len(emb.calls[0]) == 5

def test_embeds_title_plus_text(session):
    _seed(session, n=1)
    emb = FakeEmbedder()
    embed(session, emb)
    assert emb.calls[0][0].startswith("T0")
    assert "body 0" in emb.calls[0][0]

def test_backend_failure_marks_the_batch_failed_not_the_process(session):
    _seed(session)
    res = embed(session, ExplodingEmbedder())
    assert res.failed == 3 and res.processed == 0
    assert all(i.stage is Stage.FAILED for i in session.query(Item).all())
    assert "gpu on fire" in session.query(Item).first().error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embed_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.stages.embed'`

- [ ] **Step 3: Implement `feed/stages/embed.py`**

```python
# feed/stages/embed.py
from __future__ import annotations
import logging
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.embedding.base import Embedder, pack
from feed.models import Item, Stage
from feed.stages.base import StageResult

log = logging.getLogger(__name__)

def embed_text_for(item: Item) -> str:
    """Title carries most of the story identity; text disambiguates.

    Both models truncate (256 tokens for MiniLM, 512 for bge), so putting the
    title first guarantees the most identifying text survives truncation.
    """
    return f"{item.title}\n\n{item.text or ''}".strip()

def embed(session: Session, embedder: Embedder, limit: int = 256) -> StageResult:
    result = StageResult(name="embed")
    stmt = (select(Item).where(Item.stage == Stage.NORMALIZED)
            .order_by(Item.id).limit(limit))
    items = list(session.scalars(stmt))
    if not items:
        return result

    try:
        vectors = embedder.encode([embed_text_for(i) for i in items])
    except Exception as exc:
        session.rollback()
        for item in items:
            item.stage = Stage.FAILED
            item.error = f"embedding failed: {type(exc).__name__}: {exc}"
        session.commit()
        result.failed = len(items)
        result.errors = [(i.id, str(exc)) for i in items]
        log.error("embed batch of %d failed: %s", len(items), exc)
        return result

    for item, vec in zip(items, vectors):
        item.embedding = pack(vec)
        item.embedding_model_id = embedder.model_id
        item.stage = Stage.EMBEDDED
        item.error = None
    session.commit()
    result.processed = len(items)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embed_stage.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add feed/stages/embed.py tests/test_embed_stage.py
git commit -m "feat: batched embed stage recording model id per vector"
```

---

### Task 10: Entity extraction and clustering signals

**Files:**
- Create: `feed/clustering/__init__.py`, `feed/clustering/entities.py`, `feed/clustering/signals.py`
- Test: `tests/test_clustering_signals.py`

**Interfaces:**
- Consumes: `unpack` (Task 8)
- Produces: `extract_entities(text: str) -> set[str]`; `cosine(a, b) -> float`; `entity_overlap(a: set[str], b: set[str]) -> float`; `link_overlap(a: list[str], b: list[str]) -> float`; `time_proximity(a: datetime, b: datetime, window_hours: int) -> float`; `blend(cos: float, ent: float, *, cosine_weight: float, entity_weight: float) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering_signals.py
from datetime import datetime, timedelta, timezone
import numpy as np
from feed.clustering.entities import extract_entities
from feed.clustering.signals import (blend, cosine, entity_overlap, link_overlap,
                                     time_proximity)

def test_extract_entities_finds_orgs_and_model_names():
    ents = extract_entities("DeepSeek releases V4, an open-weights MoE model.")
    assert "deepseek" in ents
    assert "v4" in ents

def test_extract_entities_drops_leading_stopwords():
    ents = extract_entities("The Commission postponed the deadline.")
    assert "the" not in ents
    assert "commission" in ents

def test_extract_entities_is_case_insensitive_in_output():
    assert extract_entities("NVIDIA beats estimates") == extract_entities("Nvidia beats estimates")

def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(v, v) == 1.0

def test_cosine_handles_zero_vector_without_dividing_by_zero():
    z = np.zeros(3, dtype=np.float32)
    assert cosine(z, np.array([1.0, 0, 0], dtype=np.float32)) == 0.0

def test_entity_overlap_is_jaccard():
    assert entity_overlap({"a", "b"}, {"a", "b"}) == 1.0
    assert entity_overlap({"a", "b"}, {"b", "c"}) == 1 / 3
    assert entity_overlap(set(), set()) == 0.0

def test_link_overlap_detects_shared_source_document():
    a = ["https://huggingface.co/deepseek/v4", "https://x.com/a"]
    b = ["https://huggingface.co/deepseek/v4"]
    assert link_overlap(a, b) > 0.0
    assert link_overlap(a, ["https://unrelated.example"]) == 0.0

def test_time_proximity_decays_to_zero_at_window_edge():
    t = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)
    assert time_proximity(t, t, window_hours=48) == 1.0
    assert time_proximity(t, t + timedelta(hours=48), window_hours=48) == 0.0
    mid = time_proximity(t, t + timedelta(hours=24), window_hours=48)
    assert 0.4 < mid < 0.6

def test_blend_matches_the_measured_weights():
    # spec Appendix A: 0.6*cosine + 0.4*entities
    assert blend(1.0, 0.0, cosine_weight=0.6, entity_weight=0.4) == 0.6
    assert blend(0.0, 1.0, cosine_weight=0.6, entity_weight=0.4) == 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clustering_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.clustering'`

- [ ] **Step 3: Implement `feed/clustering/entities.py`**

```python
# feed/clustering/entities.py
from __future__ import annotations
import re

# Sentence-initial and generic capitalised words that carry no identity.
STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "we", "our", "its",
    "in", "on", "at", "for", "and", "but", "new", "it", "they", "he", "she",
    "as", "by", "with", "from", "to", "of", "is", "are", "was", "were",
}

_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z0-9.\-]{2,}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_VERSIONED = re.compile(r"\b[A-Za-z]+-?\d+(?:\.\d+)?\b")

def extract_entities(text: str) -> set[str]:
    """Cheap proxy for named entities: capitalised tokens, acronyms, and
    versioned names like 'V4' or 'GPT-5'.

    Deliberately not a NER model. Spec §3.4 measured this crude signal as
    enough to flip the clustering separation margin positive when blended
    with cosine, and it costs nothing.
    """
    if not text:
        return set()
    found: set[str] = set()
    for pattern in (_CAPITALISED, _ACRONYM, _VERSIONED):
        found |= {m.group(0).lower() for m in pattern.finditer(text)}
    return {w for w in found if w not in STOPWORDS and len(w) > 1}
```

- [ ] **Step 4: Implement `feed/clustering/signals.py`**

```python
# feed/clustering/signals.py
from __future__ import annotations
from datetime import datetime
from urllib.parse import urlsplit
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def entity_overlap(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)

def _key(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}"

def link_overlap(a: list[str], b: list[str]) -> float:
    """Two articles citing the same primary document are very likely the same
    story. Compared on host+path so query strings do not defeat the match."""
    sa = {_key(u) for u in (a or [])}
    sb = {_key(u) for u in (b or [])}
    return entity_overlap(sa, sb)

def time_proximity(a: datetime, b: datetime, window_hours: int) -> float:
    gap = abs((a - b).total_seconds()) / 3600.0
    if gap >= window_hours:
        return 0.0
    return 1.0 - (gap / window_hours)

def blend(cos: float, ent: float, *, cosine_weight: float, entity_weight: float) -> float:
    return cosine_weight * cos + entity_weight * ent
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clustering_signals.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add feed/clustering/ tests/test_clustering_signals.py
git commit -m "feat: entity extraction and clustering similarity signals"
```

---

### Task 11: Cluster stage with the adjudicator seam

**Files:**
- Create: `feed/clustering/adjudicate.py`, `feed/stages/cluster.py`
- Test: `tests/test_cluster_stage.py`

**Interfaces:**
- Consumes: signals (Task 10), `unpack` (Task 8), `Item`/`Story`/`Stage` (Task 2), `ClusteringConfig` (Task 1)
- Produces: `Verdict` enum with `SAME`, `DIFFERENT`, `AMBIGUOUS`; `Adjudicator` protocol with `decide(pair_score: float, left: Item, right: Item) -> Verdict`; `ThresholdAdjudicator(merge_threshold, ambiguous_band)`; `NullAdjudicator` (resolves AMBIGUOUS to DIFFERENT); `cluster(session, cfg, adjudicator, now=None, limit=200) -> StageResult`

The `AMBIGUOUS` verdict is the Phase 2 seam. Phase 1 must not merge on ambiguity — a wrongly merged story is far more damaging than a split one, because a split shows up as two rows the reader can see, while a merge silently hides an event.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster_stage.py
from datetime import datetime, timedelta, timezone
import numpy as np
from feed.clustering.adjudicate import (NullAdjudicator, ThresholdAdjudicator, Verdict)
from feed.config import ClusteringConfig
from feed.embedding.base import pack
from feed.models import Item, Source, Stage, Story
from feed.stages.cluster import cluster

NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)

def _vec(*xs): return pack(np.array(xs, dtype=np.float32))

def _seed(session, rows):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.add(Source(id="t", plugin="rss", config={}, cadence_minutes=30))
    for i, (src, title, vec, offset) in enumerate(rows):
        session.add(Item(source_id=src, url=f"http://x/{i}", url_hash=f"h{i}",
                         title=title, text=title, embedding=vec,
                         embedding_model_id="fake/v1",
                         published_at=NOW + timedelta(hours=offset),
                         stage=Stage.EMBEDDED))
    session.commit()

def test_threshold_adjudicator_bands():
    adj = ThresholdAdjudicator(merge_threshold=0.50, ambiguous_band=0.06)
    assert adj.decide(0.70, None, None) is Verdict.SAME
    assert adj.decide(0.30, None, None) is Verdict.DIFFERENT
    assert adj.decide(0.52, None, None) is Verdict.AMBIGUOUS

def test_null_adjudicator_never_merges_on_ambiguity():
    adj = NullAdjudicator(ThresholdAdjudicator(0.50, 0.06))
    assert adj.decide(0.52, None, None) is Verdict.DIFFERENT
    assert adj.decide(0.90, None, None) is Verdict.SAME

def test_similar_items_from_two_outlets_become_one_story(session):
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), 1),
    ])
    res = cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.processed == 2
    stories = session.query(Story).all()
    assert len(stories) == 1
    assert stories[0].item_count == 2
    assert stories[0].outlet_count == 2

def test_unrelated_items_stay_separate(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("t", "Texas grid capacity warning", _vec(0, 1, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2

def test_items_outside_the_time_window_do_not_merge(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("t", "DeepSeek releases V4", _vec(1, 0, 0), 200),   # far outside 48h
    ])
    cluster(session, ClusteringConfig(window_hours=48),
            NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2

def test_outlet_count_does_not_double_count_one_source(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("s", "DeepSeek V4 is out now", _vec(1, 0, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    story = session.query(Story).one()
    assert story.item_count == 2
    assert story.outlet_count == 1

def test_items_advance_to_clustered(session):
    _seed(session, [("s", "A story", _vec(1, 0, 0), 0)])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Item).one().stage is Stage.CLUSTERED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cluster_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.clustering.adjudicate'`

- [ ] **Step 3: Implement `feed/clustering/adjudicate.py`**

```python
# feed/clustering/adjudicate.py
from __future__ import annotations
import enum
from typing import Protocol

class Verdict(enum.Enum):
    SAME = "same"
    DIFFERENT = "different"
    AMBIGUOUS = "ambiguous"

class Adjudicator(Protocol):
    def decide(self, pair_score: float, left, right) -> Verdict: ...

class ThresholdAdjudicator:
    """Blended-score thresholding with an explicit uncertainty band.

    The band exists because the spike measured a *negative* separation margin
    for cosine alone: no single threshold cleanly divides same-story from
    different-story pairs. Scores inside the band are honestly reported as
    AMBIGUOUS rather than forced to a guess.
    """

    def __init__(self, merge_threshold: float = 0.50, ambiguous_band: float = 0.06):
        self.merge_threshold = merge_threshold
        self.ambiguous_band = ambiguous_band

    def decide(self, pair_score: float, left=None, right=None) -> Verdict:
        low = self.merge_threshold - self.ambiguous_band / 2
        high = self.merge_threshold + self.ambiguous_band / 2
        if pair_score >= high:
            return Verdict.SAME
        if pair_score <= low:
            return Verdict.DIFFERENT
        return Verdict.AMBIGUOUS

class NullAdjudicator:
    """Phase 1 seam. Resolves AMBIGUOUS to DIFFERENT.

    Splitting a story wrongly produces two visible rows the reader can see and
    reconcile. Merging wrongly silently hides an event, which violates success
    criterion 1. When uncertain, split. Phase 2 replaces this with an LLM
    adjudicator that actually answers the question.
    """

    def __init__(self, inner: Adjudicator):
        self.inner = inner

    def decide(self, pair_score: float, left=None, right=None) -> Verdict:
        verdict = self.inner.decide(pair_score, left, right)
        return Verdict.DIFFERENT if verdict is Verdict.AMBIGUOUS else verdict
```

- [ ] **Step 4: Implement `feed/stages/cluster.py`**

```python
# feed/stages/cluster.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.clustering.adjudicate import Adjudicator, Verdict
from feed.clustering.entities import extract_entities
from feed.clustering.signals import (blend, cosine, entity_overlap, link_overlap,
                                     time_proximity)
from feed.config import ClusteringConfig
from feed.embedding.base import pack, unpack
from feed.models import Item, Stage, Story
from feed.stages.base import StageResult

def pair_score(left: Item, right: Item, cfg: ClusteringConfig) -> float:
    """Blend the four signals from spec 3.4.

    cosine and entity overlap carry the measured weights. Shared outbound
    links nudge upward only - two articles citing the same primary document
    are strong evidence of one event, but not citing one is no evidence
    against. Time proximity decays the whole score toward the window edge, so
    a same-day match beats an otherwise identical match two days apart.
    """
    cos = cosine(unpack(left.embedding), unpack(right.embedding))
    ents = entity_overlap(
        extract_entities(f"{left.title} {left.text or ''}"),
        extract_entities(f"{right.title} {right.text or ''}"),
    )
    score = blend(cos, ents, cosine_weight=cfg.cosine_weight,
                  entity_weight=cfg.entity_weight)

    links = link_overlap(left.outbound_links or [], right.outbound_links or [])
    score = min(1.0, score + 0.10 * links)

    if left.published_at and right.published_at:
        score *= time_proximity(left.published_at, right.published_at,
                                cfg.window_hours)
    return score

def cluster(session: Session, cfg: ClusteringConfig, adjudicator: Adjudicator,
            *, now: datetime | None = None, limit: int = 200) -> StageResult:
    now = now or datetime.now(timezone.utc)
    result = StageResult(name="cluster")
    cutoff = now - timedelta(hours=cfg.window_hours)

    pending = list(session.scalars(
        select(Item).where(Item.stage == Stage.EMBEDDED).order_by(Item.id).limit(limit)
    ))

    for item in pending:
        try:
            candidates = list(session.scalars(
                select(Item).where(
                    Item.story_id.is_not(None),
                    Item.stage == Stage.CLUSTERED,
                    Item.published_at >= cutoff,
                    Item.embedding_model_id == item.embedding_model_id,
                )
            ))
            best_story_id, best = None, 0.0
            for other in candidates:
                if item.published_at and other.published_at:
                    gap = abs((item.published_at - other.published_at).total_seconds())
                    if gap > cfg.window_hours * 3600:
                        continue
                s = pair_score(item, other, cfg)
                if s > best:
                    best, best_story_id = s, other.story_id

            verdict = (adjudicator.decide(best, item, None)
                       if best_story_id is not None else Verdict.DIFFERENT)

            if verdict is Verdict.SAME and best_story_id is not None:
                story = session.get(Story, best_story_id)
            else:
                story = Story(title=item.title, first_seen=item.published_at or now,
                              updated_at=item.published_at or now, item_count=0)
                session.add(story)
                session.flush()

            item.story_id = story.id
            item.stage = Stage.CLUSTERED
            item.error = None
            session.flush()

            members = list(session.scalars(select(Item).where(Item.story_id == story.id)))
            story.item_count = len(members)
            story.outlet_count = len({m.source_id for m in members})
            story.updated_at = max(m.published_at or now for m in members)
            vectors = np.array([unpack(m.embedding) for m in members], dtype=np.float32)
            story.centroid = pack(vectors.mean(axis=0))
            session.commit()
            result.processed += 1
        except Exception as exc:
            session.rollback()
            fresh = session.get(Item, item.id)
            if fresh is not None:
                fresh.stage = Stage.FAILED
                fresh.error = f"cluster: {type(exc).__name__}: {exc}"
                session.commit()
            result.failed += 1
            result.errors.append((item.id, str(exc)))

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cluster_stage.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add feed/clustering/adjudicate.py feed/stages/cluster.py tests/test_cluster_stage.py
git commit -m "feat: cluster stage with explicit ambiguity band and null adjudicator"
```

---

### Task 12: Golden-set clustering regression test

**Files:**
- Create: `tests/golden/__init__.py`, `tests/golden/corpus.py`, `tests/golden/test_golden.py`
- Test: itself

**Interfaces:**
- Consumes: `build_embedder` (Task 8), signals (Task 10)
- Produces: `CORPUS: list[tuple[str, str, str]]` as `(story_label, outlet, text)`; `safe_band_width(scores, labels) -> tuple[float, float, float]` returning `(low, high, width)`

Spec §6 requires this test. It is the only guard on the fragile part of the system: cosine alone gave a 0.02-wide safe band, the blend gave 0.10. A regression that narrows the band would silently wreck the feed and no other test would notice.

- [ ] **Step 1: Write the corpus**

```python
# tests/golden/corpus.py
"""Hand-labelled clustering corpus. (story_label, outlet, headline + lead).

Labels A-D are multi-outlet stories; S* are singletons. Grown over time from
real misclusterings observed in production — every fix should add its case.
"""

CORPUS: list[tuple[str, str, str]] = [
    ("A", "TechCrunch", "DeepSeek releases V4, an open-weights MoE model it claims matches frontier systems. The Chinese lab published weights under an MIT license on Hugging Face early Tuesday."),
    ("A", "TheVerge", "DeepSeek's new V4 model is free to download and claims GPT-class performance. The release lands under a permissive license, unusual for a model of this scale."),
    ("A", "VentureBeat", "Chinese AI lab DeepSeek open-sources V4 mixture-of-experts model with 1.2T parameters. Benchmarks published alongside the release claim parity with closed frontier models on coding tasks."),
    ("A", "Reuters", "DeepSeek publishes new AI model weights publicly, escalating open-source competition. The move intensifies pressure on US labs that keep their frontier weights private."),
    ("A", "HackerNews", "Show HN: DeepSeek V4 weights are up on HuggingFace. Running it locally on 2x4090 with 4-bit quantization, quality seems genuinely close to Opus for refactoring work."),
    ("A", "ArsTechnica", "DeepSeek V4 arrives with open weights and bold benchmark claims. Independent evaluation has not yet confirmed the lab's reported SWE-bench numbers."),
    ("B", "Politico", "EU delays enforcement of AI Act high-risk provisions by eighteen months. The Commission cited unfinished technical standards as the reason for the postponement."),
    ("B", "Reuters", "European Commission postpones key AI Act obligations until 2028. Industry groups had lobbied heavily for additional implementation time."),
    ("B", "EURACTIV", "Brussels pushes back AI Act Article 6 deadline amid standards delay. Civil society organisations criticised the decision as capitulation to industry pressure."),
    ("B", "FT", "Brussels grants AI companies extra time to comply with landmark rules. The delay affects obligations for systems classified as high-risk under the regulation."),
    ("C", "CNBC", "Nvidia beats Q3 estimates as datacenter revenue climbs 62% year over year. The company guided above consensus for the coming quarter."),
    ("C", "Bloomberg", "Nvidia's datacenter sales surge again, topping analyst expectations. Shares rose in after-hours trading following the report."),
    ("C", "WSJ", "Nvidia quarterly results exceed forecasts on continued AI infrastructure demand. Executives said supply constraints are easing but remain a factor."),
    ("D", "arXiv", "Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents. We introduce a learned consolidator that rewrites regions of agent memory during idle compute."),
    ("D", "TwitterThread", "New paper: Auto-Dreamer does offline memory consolidation for LLM agents. Treats a memory region as read-only evidence then synthesizes a compact replacement. Nice results on long-horizon tasks."),
    ("D", "MarkTechPost", "Researchers propose Auto-Dreamer for agent memory consolidation during idle time. The method abstracts across sessions to replace bloated memory regions with compact summaries."),
    ("S1", "Anthropic", "Introducing improvements to Claude Code hooks and plugin configuration. Teams can now define lifecycle hooks that run before and after tool calls."),
    ("S2", "IEEE", "Autonomous chemistry lab at Argonne runs 24-hour discovery loops for battery materials. The facility combines robotic synthesis with model-driven hypothesis generation."),
    ("S3", "TheInformation", "OpenAI in talks to raise at a higher valuation, sources say. The round would value the company well above its previous mark."),
    ("S4", "arXiv", "Sparse autoencoders fail to recover ground-truth features in synthetic settings. We construct toy models where the true features are known and evaluate recovery rates."),
    ("S5", "DataCenterDynamics", "Texas grid operator warns of capacity shortfall from datacenter buildout. Interconnection queues have grown substantially over the past eighteen months."),
    ("S6", "GitHub", "Release v2.0 of a popular vector database adds hybrid search and metadata filtering. The update focuses on recall improvements for mixed keyword and semantic queries."),
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/golden/test_golden.py
import numpy as np
import pytest
from feed.clustering.entities import extract_entities
from feed.clustering.signals import blend, cosine, entity_overlap
from feed.config import EmbeddingConfig
from feed.embedding import build_embedder
from tests.golden.corpus import CORPUS

MIN_BAND_WIDTH = 0.06   # measured 0.10 for the blend; 0.02 for cosine alone

def _union_find_clusters(scores, n, threshold):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if scores[i][j] >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return {frozenset(v) for v in groups.values()}

@pytest.fixture(scope="module")
def scores():
    labels = [c[0] for c in CORPUS]
    texts = [c[2] for c in CORPUS]
    cfg = EmbeddingConfig(backend="onnx",
                          model="sentence-transformers/all-MiniLM-L6-v2",
                          device="cpu", batch_size=32)
    V = build_embedder(cfg).encode(texts)
    ents = [extract_entities(t) for t in texts]
    n = len(texts)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = blend(cosine(V[i], V[j]), entity_overlap(ents[i], ents[j]),
                            cosine_weight=0.6, entity_weight=0.4)
    return labels, M, n

@pytest.mark.slow
def test_blend_recovers_ground_truth_over_a_usable_band(scores):
    labels, M, n = scores
    truth = {}
    for i, l in enumerate(labels):
        truth.setdefault(l, []).append(i)
    truth_sets = {frozenset(v) for v in truth.values()}

    working = [round(t, 2) for t in np.arange(0.20, 0.95, 0.01)
               if _union_find_clusters(M, n, round(t, 2)) == truth_sets]

    assert working, (
        "No threshold recovers the labelled stories. The clustering signals "
        "have regressed below usability."
    )
    low, high = min(working), max(working)
    width = high - low
    assert width >= MIN_BAND_WIDTH, (
        f"Safe threshold band narrowed to {width:.2f} (band {low:.2f}-{high:.2f}); "
        f"minimum is {MIN_BAND_WIDTH}. Clustering is now fragile even though it "
        f"still passes at some threshold."
    )

@pytest.mark.slow
def test_configured_threshold_sits_inside_the_working_band(scores):
    from feed.config import ClusteringConfig
    labels, M, n = scores
    truth = {}
    for i, l in enumerate(labels):
        truth.setdefault(l, []).append(i)
    truth_sets = {frozenset(v) for v in truth.values()}
    configured = ClusteringConfig().merge_threshold
    assert _union_find_clusters(M, n, configured) == truth_sets, (
        f"configured merge_threshold={configured} does not recover ground truth"
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/golden/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.golden.corpus'` before Step 1, or an assertion failure if the default `merge_threshold` is outside the measured band.

- [ ] **Step 4: Reconcile the default threshold with the measured band**

If `test_configured_threshold_sits_inside_the_working_band` fails, the default in `feed/config.py` is wrong for the CPU model. Print the band and set the default to its midpoint:

```bash
.venv/Scripts/python.exe -m pytest tests/golden/test_golden.py::test_blend_recovers_ground_truth_over_a_usable_band -v -s
```

Update `ClusteringConfig.merge_threshold` and `feed.toml` to the midpoint of the reported band. Record the observed band in a comment beside the default.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/golden/ -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add tests/golden/ feed/config.py feed.toml
git commit -m "test: golden-set clustering regression guarding threshold band width"
```

---

### Task 13: Importance scoring

**Files:**
- Create: `feed/scoring/__init__.py`, `feed/scoring/signals.py`, `feed/scoring/combine.py`, `feed/stages/score.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Story`/`Item`/`Source` (Task 2), `unpack` (Task 8), `cosine` (Task 10), `ScoringConfig` (Task 1)
- Produces: `authority(session, story) -> float`; `velocity(story) -> float`; `novelty(session, story, days=90) -> float`; `entity_weight(session, story) -> float`; `combine(parts: dict[str, float], weights: dict[str, float]) -> float`; `score_stories(session, cfg, now=None) -> StageResult`

Per spec §3.6 this produces reader-independent importance only. Personal fit is computed on the client and must not appear here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from feed.config import ScoringConfig
from feed.embedding.base import pack
from feed.models import Item, Source, Stage, Story
from feed.scoring.combine import combine
from feed.scoring.signals import authority, novelty, velocity
from feed.stages.score import score_stories

NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)

def _story(session, *, outlets, sources_authority=0.5, vec=(1.0, 0.0), age_h=1):
    for i, name in enumerate(outlets):
        if session.get(Source, name) is None:
            session.add(Source(id=name, plugin="rss", config={},
                               cadence_minutes=30, authority=sources_authority))
    st = Story(title="S", first_seen=NOW - timedelta(hours=age_h),
               updated_at=NOW - timedelta(hours=age_h),
               item_count=len(outlets), outlet_count=len(set(outlets)),
               centroid=pack(np.array(vec, dtype=np.float32)))
    session.add(st)
    session.flush()
    for i, name in enumerate(outlets):
        session.add(Item(source_id=name, url=f"http://x/{name}/{i}",
                         url_hash=f"{name}{i}", title="t", text="t",
                         embedding=pack(np.array(vec, dtype=np.float32)),
                         embedding_model_id="fake/v1", story_id=st.id,
                         published_at=NOW - timedelta(hours=age_h),
                         stage=Stage.CLUSTERED))
    session.commit()
    return st

def test_velocity_rises_with_independent_outlets(session):
    one = _story(session, outlets=["a"])
    many = _story(session, outlets=["b", "c", "d", "e", "f"], vec=(0.0, 1.0))
    assert velocity(many) > velocity(one)

def test_velocity_counts_outlets_not_articles(session):
    syndicated = _story(session, outlets=["a", "a", "a", "a"])
    genuine = _story(session, outlets=["b", "c", "d", "e"], vec=(0.0, 1.0))
    assert velocity(genuine) > velocity(syndicated)

def test_authority_averages_contributing_sources(session):
    st = _story(session, outlets=["hi"], sources_authority=0.9)
    assert authority(session, st) == pytest.approx(0.9)

def test_novelty_is_low_for_a_near_duplicate_of_an_older_story(session):
    _story(session, outlets=["a"], vec=(1.0, 0.0), age_h=200)
    follow_up = _story(session, outlets=["b"], vec=(1.0, 0.0), age_h=1)
    fresh = _story(session, outlets=["c"], vec=(0.0, 1.0), age_h=1)
    assert novelty(session, follow_up) < novelty(session, fresh)

def test_combine_is_a_weighted_sum_clamped_to_unit_range():
    parts = {"authority": 1.0, "velocity": 1.0, "novelty": 1.0, "entity": 1.0}
    weights = {"authority": 0.25, "velocity": 0.4, "novelty": 0.2, "entity": 0.15}
    assert combine(parts, weights) == pytest.approx(1.0)
    assert combine({"authority": 0.0}, {"authority": 1.0}) == 0.0

def test_combine_ignores_weights_with_no_matching_part():
    assert combine({"velocity": 1.0}, {"velocity": 0.5, "missing": 0.5}) == pytest.approx(0.5)

def test_score_stories_persists_score_and_breakdown(session):
    _story(session, outlets=["a", "b", "c"])
    res = score_stories(session, ScoringConfig(), now=NOW)
    assert res.processed == 1
    st = session.query(Story).one()
    assert st.score is not None and 0.0 <= st.score <= 1.0
    assert set(st.score_breakdown) == {"authority", "velocity", "novelty", "entity"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.scoring'`

- [ ] **Step 3: Implement `feed/scoring/signals.py`**

```python
# feed/scoring/signals.py
from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.clustering.signals import cosine
from feed.embedding.base import unpack
from feed.models import Item, Source, Story

def authority(session: Session, story: Story) -> float:
    rows = session.scalars(
        select(Source.authority).join(Item, Item.source_id == Source.id)
        .where(Item.story_id == story.id)
    ).all()
    return float(sum(rows) / len(rows)) if rows else 0.5

def velocity(story: Story) -> float:
    """Independent outlets, log-compressed. Counting outlets rather than
    articles is what stops one publisher's syndication network from
    manufacturing importance."""
    return min(1.0, math.log1p(max(0, story.outlet_count)) / math.log1p(10))

def novelty(session: Session, story: Story, days: int = 90) -> float:
    if story.centroid is None:
        return 1.0
    cutoff = (story.first_seen or datetime.now(timezone.utc)) - timedelta(days=days)
    others = session.scalars(
        select(Story).where(Story.id != story.id, Story.first_seen >= cutoff,
                            Story.first_seen < story.first_seen,
                            Story.centroid.is_not(None))
    ).all()
    if not others:
        return 1.0
    mine = unpack(story.centroid)
    peak = max(cosine(mine, unpack(o.centroid)) for o in others)
    return float(max(0.0, 1.0 - peak))

def entity_weight(session: Session, story: Story) -> float:
    from feed.models import Entity, StoryEntity
    rows = session.scalars(
        select(Entity.weight).join(StoryEntity, StoryEntity.entity_id == Entity.id)
        .where(StoryEntity.story_id == story.id)
    ).all()
    return float(max(rows)) if rows else 0.5
```

- [ ] **Step 4: Implement `feed/scoring/combine.py` and `feed/stages/score.py`**

```python
# feed/scoring/combine.py
from __future__ import annotations

def combine(parts: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the signals actually present, clamped to [0, 1].

    Renormalising over present signals means a missing signal degrades the
    score's precision, never its scale.
    """
    used = {k: w for k, w in weights.items() if k in parts}
    total = sum(used.values())
    if total == 0:
        return 0.0
    raw = sum(parts[k] * w for k, w in used.items()) / total
    return max(0.0, min(1.0, raw))
```

```python
# feed/stages/score.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.config import ScoringConfig
from feed.models import Item, Stage, Story
from feed.scoring.combine import combine
from feed.scoring.signals import authority, entity_weight, novelty, velocity
from feed.stages.base import StageResult

def score_stories(session: Session, cfg: ScoringConfig, *,
                  now: datetime | None = None) -> StageResult:
    now = now or datetime.now(timezone.utc)
    result = StageResult(name="score")

    story_ids = session.scalars(
        select(Item.story_id).where(Item.stage == Stage.CLUSTERED).distinct()
    ).all()

    for story_id in story_ids:
        story = session.get(Story, story_id)
        if story is None:
            continue
        try:
            parts = {
                "authority": authority(session, story),
                "velocity": velocity(story),
                "novelty": novelty(session, story),
                "entity": entity_weight(session, story),
            }
            story.score = combine(parts, cfg.weights)
            story.score_breakdown = parts
            for item in session.scalars(
                select(Item).where(Item.story_id == story.id,
                                   Item.stage == Stage.CLUSTERED)
            ):
                item.stage = Stage.SCORED
            session.commit()
            result.processed += 1
        except Exception as exc:
            session.rollback()
            result.failed += 1
            result.errors.append((story_id, str(exc)))

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scoring.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add feed/scoring/ feed/stages/score.py tests/test_scoring.py
git commit -m "feat: reader-independent importance scoring"
```

---

### Task 14: CLI and end-to-end run

**Files:**
- Create: `feed/cli.py`, `feed/__main__.py`, `sources.example.toml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above
- Produces: `main(argv: list[str] | None = None) -> int`; subcommands `init`, `sources add`, `sources list`, `run`, `stats`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path
from feed.cli import main

FIX = Path(__file__).parent / "fixtures" / "sample_rss.xml"

def _cfg(tmp_path) -> Path:
    p = tmp_path / "feed.toml"
    p.write_text(
        f'[database]\nurl = "sqlite:///{(tmp_path / "t.db").as_posix()}"\n'
        '[embedding]\nbackend = "onnx"\n'
        'model = "sentence-transformers/all-MiniLM-L6-v2"\n'
        'device = "cpu"\nbatch_size = 8\n',
        encoding="utf-8",
    )
    return p

def test_init_creates_the_database(tmp_path):
    cfg = _cfg(tmp_path)
    assert main(["--config", str(cfg), "init"]) == 0
    assert (tmp_path / "t.db").exists()

def test_sources_add_then_list(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "add", "--id", "rss:example",
               "--plugin", "rss", "--config-json", f'{{"path": "{FIX.as_posix()}"}}'])
    assert rc == 0
    main(["--config", str(cfg), "sources", "list"])
    assert "rss:example" in capsys.readouterr().out

def test_unknown_plugin_is_rejected_at_add_time(tmp_path):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "add", "--id", "x",
               "--plugin", "nope", "--config-json", "{}"])
    assert rc == 2

def test_full_run_produces_scored_stories(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    main(["--config", str(cfg), "sources", "add", "--id", "rss:example",
          "--plugin", "rss", "--config-json", f'{{"path": "{FIX.as_posix()}"}}'])
    assert main(["--config", str(cfg), "run"]) == 0
    main(["--config", str(cfg), "stats"])
    out = capsys.readouterr().out
    assert "stories" in out
```

Mark the last test `@pytest.mark.slow` — it downloads the MiniLM model on first run.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'feed.cli'`

- [ ] **Step 3: Implement `feed/cli.py`**

```python
# feed/cli.py
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from sqlalchemy import func, select
import feed.sources  # noqa: F401  (registers plugins)
from feed.clustering.adjudicate import NullAdjudicator, ThresholdAdjudicator
from feed.config import load_config
from feed.db import create_all, make_engine, make_session_factory
from feed.embedding import build_embedder
from feed.embedding.resolve import resolve
from feed.models import Item, Source, Stage, Story
from feed.sources.registry import known_plugins
from feed.stages.cluster import cluster
from feed.stages.collect import collect
from feed.stages.embed import embed
from feed.stages.normalize import normalize
from feed.stages.score import score_stories

def _session(cfg):
    engine = make_engine(cfg.database.url)
    return engine, make_session_factory(engine)

def cmd_init(args, cfg) -> int:
    engine, _ = _session(cfg)
    create_all(engine)
    print(f"initialised {cfg.database.url}")
    return 0

def cmd_sources_add(args, cfg) -> int:
    if args.plugin not in known_plugins():
        print(f"unknown plugin {args.plugin!r}; known: {known_plugins()}", file=sys.stderr)
        return 2
    try:
        conf = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        print(f"--config-json is not valid JSON: {exc}", file=sys.stderr)
        return 2
    _, factory = _session(cfg)
    with factory() as s:
        s.merge(Source(id=args.id, plugin=args.plugin, config=conf,
                       cadence_minutes=args.cadence, authority=args.authority))
        s.commit()
    print(f"added source {args.id}")
    return 0

def cmd_sources_list(args, cfg) -> int:
    _, factory = _session(cfg)
    with factory() as s:
        for src in s.scalars(select(Source).order_by(Source.id)):
            state = "ok" if src.consecutive_failures == 0 else f"FAILING x{src.consecutive_failures}"
            print(f"{src.id:<28} {src.plugin:<18} every {src.cadence_minutes:>4}m  {state}")
    return 0

def cmd_run(args, cfg) -> int:
    _, factory = _session(cfg)
    backend, model, device = resolve(cfg.embedding)
    print(f"embedding: backend={backend} model={model} device={device}")
    embedder = build_embedder(cfg.embedding)
    adjudicator = NullAdjudicator(
        ThresholdAdjudicator(merge_threshold=cfg.clustering.merge_threshold)
    )
    with factory() as s:
        c = collect(s)
        print(f"collect:   new={c.new_items} dupes={c.skipped_duplicates} "
              f"source_errors={len(c.source_errors)}")
        for name, res in [
            ("normalize", normalize(s)),
            ("embed", embed(s, embedder, limit=cfg.embedding.batch_size)),
            ("cluster", cluster(s, cfg.clustering, adjudicator)),
            ("score", score_stories(s, cfg.scoring)),
        ]:
            print(f"{name+':':<11}ok={res.processed} failed={res.failed}")
    return 0

def cmd_stats(args, cfg) -> int:
    _, factory = _session(cfg)
    with factory() as s:
        print("items by stage:")
        for stage, n in s.execute(
            select(Item.stage, func.count()).group_by(Item.stage)
        ):
            print(f"  {stage.value:<12}{n}")
        total = s.scalar(select(func.count()).select_from(Story)) or 0
        print(f"stories: {total}")
        print("top stories by importance:")
        for st in s.scalars(
            select(Story).where(Story.score.is_not(None))
            .order_by(Story.score.desc()).limit(10)
        ):
            print(f"  {st.score:.3f}  [{st.outlet_count} outlets]  {st.title[:70]}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="feed")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("stats").set_defaults(func=cmd_stats)

    srcs = sub.add_parser("sources").add_subparsers(dest="sub", required=True)
    add = srcs.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--plugin", required=True)
    add.add_argument("--config-json", default="{}")
    add.add_argument("--cadence", type=int, default=30)
    add.add_argument("--authority", type=float, default=0.5)
    add.set_defaults(func=cmd_sources_add)
    srcs.add_parser("list").set_defaults(func=cmd_sources_list)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args, load_config(args.config))
```

```python
# feed/__main__.py
import sys
from feed.cli import main
sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the starter source list**

```toml
# sources.example.toml — feed the ids to `feed sources add`
# arxiv:ai          plugin=arxiv            {"categories": ["cs.AI","cs.CL","cs.LG"]}
# hn                plugin=hackernews       {"min_score": 150}
# gh:vllm           plugin=github_releases  {"repo": "vllm-project/vllm"}
# gh:transformers   plugin=github_releases  {"repo": "huggingface/transformers"}
# anthropic         plugin=rss  {"url": "https://www.anthropic.com/news/rss.xml"}
# openai            plugin=rss  {"url": "https://openai.com/blog/rss.xml"}
# deepmind          plugin=rss  {"url": "https://deepmind.google/blog/rss.xml"}
# hf-blog           plugin=rss  {"url": "https://huggingface.co/blog/feed.xml"}
```

- [ ] **Step 6: Run the pipeline against real sources and inspect in SQL**

```bash
.venv/Scripts/python.exe -m feed init
.venv/Scripts/python.exe -m feed sources add --id arxiv:ai --plugin arxiv \
  --config-json "{\"categories\": [\"cs.AI\", \"cs.CL\"]}" --cadence 180 --authority 0.8
.venv/Scripts/python.exe -m feed sources add --id hn --plugin hackernews \
  --config-json "{\"min_score\": 150}" --cadence 30 --authority 0.5
.venv/Scripts/python.exe -m feed run
.venv/Scripts/python.exe -m feed stats
```

Then inspect the clusters by hand — this is the phase's actual deliverable:

```sql
SELECT s.id, s.score, s.outlet_count, s.item_count, s.title
FROM story s ORDER BY s.score DESC LIMIT 20;

SELECT s.title AS story, i.source_id, i.title AS item
FROM story s JOIN item i ON i.story_id = s.id
WHERE s.item_count > 1 ORDER BY s.id;
```

Record in the plan's closing notes: how many stories were wrongly merged, how many wrongly split, and the observed distribution of `story.score`. **Open questions 5, 6 and 7 in the spec are answered from this run**, not guessed.

- [ ] **Step 7: Commit**

```bash
git add feed/cli.py feed/__main__.py sources.example.toml tests/test_cli.py
git commit -m "feat: CLI with init, sources, run, and stats"
```

---

## Phase 1 exit criteria

- [ ] `pytest` green, including the golden-set band-width test
- [ ] `feed run` completes against at least 5 real sources with zero unhandled exceptions
- [ ] `feed stats` shows stories with more than one outlet, i.e. clustering is actually merging
- [ ] Manual SQL review of the top 20 stories done, with merge and split errors counted
- [ ] Spec open questions 5 (time window), 6 (cut points) and 7 (signal weights) answered from observed data and written back into the spec

## Notes for the executor

- **Do not add LLM calls.** If clustering quality looks poor, record it and move on. Fixing it with an LLM adjudicator is Phase 2 and the seam is already in place.
- **Do not add a web server, API, or publishing step.** Phase 1 ends at SQLite.
- **When uncertain whether two items are one story, split.** A wrong split is visible to the reader; a wrong merge silently hides an event.
- The `slow` marker exists so `pytest -m "not slow"` stays fast during development. CI runs the full set.
