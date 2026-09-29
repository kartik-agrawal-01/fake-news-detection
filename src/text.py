"""Small, dependency-free text helpers (no NLTK downloads needed)."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_headline(text: str) -> str:
    """Key used to group exact-duplicate headlines: lower-cased, whitespace collapsed."""
    return " ".join(str(text).lower().split())


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without English stopwords. Used by the BM25 evidence index."""
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]
