from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .example import Example
from .lazy_example_retriever import LazyExampleRetriever


class CoverageExampleRetriever(LazyExampleRetriever):
    # Benchmarked as a set; the measured gain says nothing about other values.
    _ANCHORS = 20
    _CANDIDATE_MULTIPLE = 4
    _MIN_CANDIDATES = 100
    _RELEVANCE_WEIGHT = 0.7

    def __init__(self, examples: List[Example]) -> None:
        super().__init__(examples)

    def _fit(self) -> None:
        # TfidfVectorizer rejects a corpus with nothing to put in its vocabulary.
        sources = [ex.source for ex in self.examples]
        if not any(source.strip() for source in sources):
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        self._matrix = self._vectorizer.fit_transform(sources)

    def _rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]:
        if self._vectorizer is None or self._matrix is None:
            return []
        vector = self._vectorizer.transform([query])
        similarities = linear_kernel(vector, self._matrix).ravel()
        ranked = similarities.argsort()[::-1]
        if exclude_index is not None:
            ranked = ranked[ranked != exclude_index]
        k = min(k, len(ranked))
        if k == 0:
            return []
        if vector.nnz == 0:  # pyright: ignore[reportAttributeAccessIssue]
            return ranked[:k].tolist()
        candidates = self._distinct_candidates(
            ranked, k, None if exclude_index is None else self.examples[exclude_index]
        )
        if len(candidates) == 0:
            return []
        chosen = self._choose_for_coverage(
            vector, self._matrix[candidates], similarities[candidates], k  # pyright: ignore[reportIndexIssue]
        )
        # Ties are broken by position so that the same corpus always yields the same prompt.
        return [
            self.examples[example_index]
            for example_index in sorted(
                (int(candidates[i]) for i in chosen), key=lambda i: (similarities[i], i), reverse=True
            )
        ]

    def _distinct_candidates(self, ranked: np.ndarray, k: int, excluded: Optional[Example]) -> np.ndarray:
        limit = max(self._CANDIDATE_MULTIPLE * k, self._MIN_CANDIDATES)
        candidates: List[int] = []
        # A copy of the excluded example would show the answer to the row it is excluded for.
        seen = set() if excluded is None else {excluded}
        for index in ranked:
            example = self.examples[index]
            if example not in seen:
                candidates.append(int(index))
                seen.add(example)
            if len(candidates) >= limit:
                break
        return np.asarray(candidates)

    def _choose_for_coverage(self, vector, candidate_matrix, similarities: np.ndarray, k: int) -> List[int]:
        features = candidate_matrix[:, vector.indices].toarray() > 0
        weights = vector.data / vector.data.sum()
        relevance = similarities / max(float(similarities.max()), 1e-12)
        # The closest few are kept outright, so that strong whole-sentence analogues survive the balancing.
        anchor_count = min(self._ANCHORS, max(1, k // 2), len(features))
        chosen = list(range(anchor_count))
        available = np.ones(len(features), dtype=bool)
        available[chosen] = False
        counts = features[chosen].sum(axis=0).astype(float)
        while len(chosen) < min(k, len(features)):
            # Each n-gram counts for less the more of the chosen examples already demonstrate it.
            gain = features @ (weights / (1.0 + counts))
            gain /= max(float(gain[available].max()), 1e-12)
            utility = self._RELEVANCE_WEIGHT * relevance + (1 - self._RELEVANCE_WEIGHT) * gain
            utility[~available] = -np.inf
            best = int(np.argmax(utility))
            chosen.append(best)
            available[best] = False
            counts += features[best]
        return chosen
