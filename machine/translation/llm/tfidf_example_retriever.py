from typing import List, cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ...tokenization.latin_word_tokenizer import LatinWordTokenizer
from .example import Example
from .top_k_example_retriever import TopKExampleRetriever


class TfidfExampleRetriever(TopKExampleRetriever):

    def __init__(self, examples: List[Example]) -> None:
        self._latin_tokenizer = LatinWordTokenizer()
        super().__init__(examples)

    def _fit(self) -> None:
        # TfidfVectorizer rejects a corpus with nothing to put in its vocabulary.
        sources = [ex.source for ex in self._examples]
        if not any(self._tokenize(source) for source in sources):
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=False, tokenizer=self._tokenize, token_pattern=None  # pyright: ignore[reportArgumentType]
        )
        self._matrix = cast(csr_matrix, self._vectorizer.fit_transform(sources))

    def _score_examples(self, query: str) -> np.ndarray:
        if self._matrix is None or self._vectorizer is None:
            return np.array([])
        vector = cast(csr_matrix, self._vectorizer.transform([query]))
        return (self._matrix @ vector.T).toarray().ravel()

    def _tokenize(self, text: str) -> List[str]:
        # Strip all punctuation-only tokens
        return [token for token in self._latin_tokenizer.tokenize(text.lower()) if any(c.isalnum() for c in token)]
