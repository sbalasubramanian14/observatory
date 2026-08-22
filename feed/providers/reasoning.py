from __future__ import annotations
import re

# Verified live: qwen/qwen3.6-27b emits visible <think>...</think> reasoning
# in its output even when not asked for it. The task brief is explicit that
# "any model may do this" -- so this strips unconditionally, for every
# provider, rather than special-casing one model name that will inevitably
# go stale. Covers the handful of tag names actually seen in the wild
# across open-weight reasoning models (DeepSeek-R1-style <think>, and the
# <reasoning>/<thinking>/<reflection> variants some OpenAI-compatible
# gateways normalise to) without trying to be a general HTML/XML stripper.
_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning|reflection)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# A block that opens but is never closed (output truncated mid-thought,
# e.g. by a token limit) would otherwise survive the paired-tag regex
# above and leak into stored summaries/analysis. Once no closing tag is
# found, treat everything from the opening tag onward as reasoning noise.
_UNCLOSED_REASONING_OPEN = re.compile(
    r"<(?:think|thinking|reasoning|reflection)>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning>/<reflection> blocks from a raw
    provider response before it is parsed or stored. Called unconditionally
    by every provider's complete() (spec requirement 3) -- never
    conditional on model name, since the model producing the reasoning
    leakage is exactly the thing that changes without notice.
    """
    if not text:
        return text
    cleaned = _REASONING_BLOCK.sub("", text)
    cleaned = _UNCLOSED_REASONING_OPEN.sub("", cleaned)
    return cleaned.strip()
