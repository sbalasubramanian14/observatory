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
