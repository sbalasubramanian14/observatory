from __future__ import annotations


def combine(parts: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum over the signals actually present, clamped to [0, 1].

    The denominator is the total of `weights` as configured, NOT the total
    of only the weights that happen to have a matching part. A missing
    signal therefore degrades the score's SCALE, not just its precision: if
    "entity" (weight 0.15) is absent from `parts`, the other three signals
    are not scaled up to fill that 15% -- a story cannot reach 1.0 without
    it. This is deliberate. Renormalising over only the present subset would
    let a story get full credit from three signals alone, silently treating
    "we don't know this signal" the same as "this signal doesn't apply" --
    which is not true for a signal that simply failed to compute. A key
    present in only one of the two dicts (a weight with no matching part,
    or a part with no matching weight) contributes nothing, in both
    directions.
    """
    total = sum(weights.values())
    if total == 0:
        return 0.0
    raw = sum(parts[k] * w for k, w in weights.items() if k in parts) / total
    return max(0.0, min(1.0, raw))
