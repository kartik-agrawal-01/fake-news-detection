"""BM25 evidence index over verified-real headlines.

The index is always built from the *training* rows of a split, so a test headline can never retrieve itself.
When retrieving evidence for a training headline, pass its duplicate group as ``exclude_group`` so it cannot
retrieve itself (or an exact duplicate) either. That keeps the evidence seen in training distributed like the
evidence seen at test time.
"""
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .text import tokenize


@dataclass
class Hit:
    text: str
    score: float


class EvidenceIndex:
    def __init__(self, texts, groups=None):
        self.texts = [str(t) for t in texts]
        self.groups = list(groups) if groups is not None else list(range(len(self.texts)))
        if len(self.groups) != len(self.texts):
            raise ValueError("texts and groups must have the same length")
        if not self.texts:
            raise ValueError("cannot build an evidence index from zero documents")
        # BM25Okapi divides by the average document length, so give empty documents a placeholder token.
        self._bm25 = BM25Okapi([tokenize(t) or ["<empty>"] for t in self.texts])

    def __len__(self):
        return len(self.texts)

    def search(self, query: str, k: int = 3, exclude_group=None) -> list[Hit]:
        """Top-k documents with a positive BM25 score, best first."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        hits = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] <= 0 or len(hits) == k:
                break
            if exclude_group is not None and self.groups[i] == exclude_group:
                continue
            hits.append(Hit(self.texts[i], float(scores[i])))
        return hits


def build_real_index(df) -> EvidenceIndex:
    """Index of the real (label 1) headlines in ``df``, which must contain only training rows."""
    real = df[df["label"] == 1]
    return EvidenceIndex(real["headline"].tolist(), real["group"].tolist())
