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
