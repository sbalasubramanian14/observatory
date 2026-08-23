# feed/catalogue.py
"""sources.catalogue.toml -- the checked-in list of every source this
pipeline is supposed to run, with plugin, config, cadence, authority
weight, and territory (spec 2's four coverage areas: research | industry
| policy | infrastructure). `feed sources sync` (feed.stages.sync)
reconciles the database against this file, so adding a source becomes
"edit a list", replacing ad-hoc `feed sources add` calls.
"""
from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

Territory = Literal["research", "industry", "policy", "infrastructure"]


class CatalogueEntry(BaseModel):
    id: str
    plugin: str
    territory: Territory
    cadence_minutes: int = Field(default=60, gt=0)
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    # Per-source override of [collect].max_backfill_days -- mirrors
    # Source.max_backfill_days / `feed sources add --max-backfill-days`.
    max_backfill_days: int | None = Field(default=None, gt=0)
    config: dict = Field(default_factory=dict)
    # Set false to keep a catalogue entry documented (so its URL, config,
    # and the reason it's disabled stay visible in the file) without
    # `feed sources sync` re-enabling it. Distinct from simply removing the
    # entry, which sync treats as "no longer wanted" (deleted if it never
    # collected anything, else disabled and left alone going forward).
    enabled: bool = True


DEFAULT_CATALOGUE_PATH = Path("sources.catalogue.toml")


def load_catalogue(path: Path | str | None = None) -> list[CatalogueEntry]:
    path = Path(path) if path is not None else DEFAULT_CATALOGUE_PATH
    if not path.exists():
        raise FileNotFoundError(f"source catalogue not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    entries = [CatalogueEntry.model_validate(e) for e in raw.get("source", [])]
    ids = [e.id for e in entries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate source id(s) in catalogue {path}: {dupes}")
    return entries
