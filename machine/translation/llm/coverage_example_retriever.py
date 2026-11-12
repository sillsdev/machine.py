from typing import List, Optional, cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .example import Example
from .example_retriever import ExampleRetriever


class CoverageExampleRetriever(ExampleRetriever):
    _ANCHORS = 20
    _CANDIDATE_MULTIPLE = 4
    _MIN_CANDIDATES = 100
    _RELEVANCE_WEIGHT = 0.7

    def __init__(self, examples: List[Example]) -> None:
        super().__init__(examples)
        seen = {}
        self._unique_example_indices = set()
        self._deduplicated_example_indices = []
        for index, example in enumerate(self._examples):
            if example not in seen:
                self._unique_example_indices.add(index)
                self._deduplicated_example_indices.append(index)
                seen[example] = index
            else:
                if seen[example] in self._unique_example_indices:
                    self._unique_example_indices.remove(seen[example])

    def _fit(self) -> None:
        # TfidfVectorizer rejects a corpus with nothing to put in its vocabulary.
        sources = [ex.source for ex in self._examples]
        if not any(source.strip() for source in sources):
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        self._matrix = cast(csr_matrix, self._vectorizer.fit_transform(sources))

    def _rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]:
        if self._vectorizer is None or self._matrix is None:
            return []
        query_vector = cast(csr_matrix, self._vectorizer.transform([query]))
        similarities = linear_kernel(query_vector, self._matrix).ravel()
        ranked = similarities.argsort()[::-1]
        if k >= ranked.shape[0]:
            return [self._examples[i] for i in ranked if i != exclude_index]
        # If exclude index points to an example which is the first occurrence
        # of a multiply attested example, then we still include that example
        if exclude_index is not None and (
            exclude_index not in self._deduplicated_example_indices or exclude_index in self._unique_example_indices
        ):
            ranked = ranked[ranked != exclude_index]
        ranked = ranked[np.isin(ranked, self._deduplicated_example_indices)]
        k = min(k, len(ranked))
        if k == 0:
            return []
        if query_vector.nnz == 0:
            return [self._examples[i] for i in ranked[:k].tolist()]
        limit = max(self._CANDIDATE_MULTIPLE * k, self._MIN_CANDIDATES)
        candidates = ranked[:limit]
        if len(candidates) == 0:
            return []
        chosen = self._choose_for_coverage(query_vector, self._matrix[candidates], similarities[candidates], k)
        return [
            self._examples[example_index]
            for example_index in sorted(
                (int(candidates[i]) for i in chosen), key=lambda i: (similarities[i], -i), reverse=True
            )
        ]

    def _choose_for_coverage(
        self, query_vector: csr_matrix, candidate_matrix: csr_matrix, similarities: np.ndarray, k: int
    ) -> List[int]:
        # Restrict the features that contribute to query coverage to those present in the query.
        features = candidate_matrix[:, query_vector.indices].toarray() > 0

        # Normalize the query weights and the candidate similarities.
        weights = query_vector.data / query_vector.data.sum()
        relevance = similarities / max(float(similarities.max()), 1e-12)

        # Always choose the first anchor_count most similar candidates.
        anchor_count = min(self._ANCHORS, max(1, k // 2), len(features))
        chosen = list(range(anchor_count))
        available = np.ones(len(features), dtype=bool)
        available[chosen] = False

        # Keep track of how many times each feature is represented across the chosen examples.
        counts = features[chosen].sum(axis=0).astype(float)

        # Greedily select additional examples by repeatedly selecting
        # the example with the highest composite relevance/coverage score
        while len(chosen) < min(k, len(features)):
            # The more the n-gram feature is important in the query, the higher the gain;
            # the more times we've already selected an example that has that feature, the lower the gain.
            relevance_weighted_coverage_gain = features @ (weights / (1.0 + counts))

            # Normalize the gains so they are comparable to the normalized relevance.
            relevance_weighted_coverage_gain /= max(float(relevance_weighted_coverage_gain[available].max()), 1e-12)

            # Create the composite utility score: a weighted sum of the gain and the normalized relevance score.
            utility = (
                self._RELEVANCE_WEIGHT * relevance + (1 - self._RELEVANCE_WEIGHT) * relevance_weighted_coverage_gain
            )

            # If the example has already been selected, set the score to -inf.
            utility[~available] = -np.inf

            # Greedily choose the highest-scoring candidate and update the available values and counts.
            best = int(np.argmax(utility))
            chosen.append(best)
            available[best] = False
            counts += features[best]

        return chosen
