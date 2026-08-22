from __future__ import annotations
from datetime import datetime
from urllib.parse import urlsplit
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, computed in float64 regardless of the input dtype.

    Deliberately NOT `dot(a, b) / (norm(a) * norm(b))`: that form takes two
    separate square roots and then multiplies them back together, and each
    sqrt rounds independently. For identical vectors that means
    `norm(a) * norm(a)` is not always bit-identical to `dot(a, a)`, so
    self-similarity can land at 0.99999994 instead of exactly 1.0 (verified
    empirically with float32 embedding-sized vectors). Squaring the norms
    via a single sqrt of the product of dot products avoids that extra
    rounding step and returns exactly 1.0 for a vector compared with itself.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom_sq = float(np.dot(a, a) * np.dot(b, b))
    if denom_sq == 0.0:
        return 0.0
    val = float(np.dot(a, b)) / (denom_sq ** 0.5)
    # Guard against tiny float overshoot past [-1, 1] from accumulated error.
    return max(-1.0, min(1.0, val))

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
