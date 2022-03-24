from itertools import islice
from random import Random
from typing import Any, Callable, Generator, Optional, Tuple

from ..tokenization.tokenizer import Tokenizer
from .corpus_view import CorpusView
from .parallel_text_row import ParallelTextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class ParallelTextCorpusView(CorpusView[ParallelTextRow]):
    def invert(self) -> "ParallelTextCorpusView":
        def _invert(row: ParallelTextRow) -> ParallelTextRow:
            return row.invert()

        return self.transform(_invert)

    def tokenize(
        self, source_tokenizer: Tokenizer[str, int, str], target_tokenizer: Optional[Tokenizer[str, int, str]] = None
    ) -> "ParallelTextCorpusView":
        if target_tokenizer is None:
            target_tokenizer = source_tokenizer

        def _tokenize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = list(source_tokenizer.tokenize(row.source_text))
            row.target_segment = list(target_tokenizer.tokenize(row.target_text))
            return row

        return self.transform(_tokenize)

    def normalize(self, normalization_form: str) -> "ParallelTextCorpusView":
        def _normalize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = normalize(normalization_form, row.source_segment)
            row.target_segment = normalize(normalization_form, row.target_segment)
            return row

        return self.transform(_normalize)

    def nfc_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFC")

    def nfd_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFD")

    def nfkc_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFKD")

    def lowercase(self) -> "ParallelTextCorpusView":
        def _lowercase(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = lowercase(row.source_segment)
            row.target_segment = lowercase(row.target_segment)
            return row

        return self.transform(_lowercase)

    def escape_spaces(self) -> "ParallelTextCorpusView":
        def _escape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = escape_spaces(row.source_segment)
            row.target_segment = escape_spaces(row.target_segment)
            return row

        return self.transform(_escape_spaces)

    def unescape_spaces(self) -> "ParallelTextCorpusView":
        def _unescape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = unescape_spaces(row.source_segment)
            row.target_segment = unescape_spaces(row.target_segment)
            return row

        return self.transform(_unescape_spaces)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, seed: Any = None
    ) -> Tuple["ParallelTextCorpusView", "ParallelTextCorpusView", int, int]:
        if percent is None and size is None:
            percent = 0.1

        corpus_size = self.count()
        if percent is not None:
            split_size = int(percent * corpus_size)
            if size is not None:
                split_size = min(split_size, size)
        else:
            assert size is not None
            split_size = size

        rand = Random()
        if seed is not None:
            rand.seed(seed)
        split_indices = set(rand.sample(range(corpus_size), min(split_size, corpus_size)))

        main_corpus = self.filter_by_index(lambda _, i: i not in split_indices)
        split_corpus = self.filter_by_index(lambda _, i: i in split_indices)

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def filter(self, predicate: Callable[[ParallelTextRow], bool]) -> "ParallelTextCorpusView":
        return FilterParallelTextCorpusView(self, lambda row, _: predicate(row))

    def filter_by_index(self, predicate: Callable[[ParallelTextRow, int], bool]) -> "ParallelTextCorpusView":
        return FilterParallelTextCorpusView(self, predicate)

    def transform(self, transform: Callable[[ParallelTextRow], ParallelTextRow]) -> "ParallelTextCorpusView":
        return TransformParallelTextCorpusView(self, transform)

    def take(self, count: int) -> "ParallelTextCorpusView":
        return TakeParallelTextCorpusView(self, count)


class TransformParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(self, corpus: ParallelTextCorpusView, transform: Callable[[ParallelTextRow], ParallelTextRow]):
        self._corpus = corpus
        self._transform = transform

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class FilterParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(self, corpus: ParallelTextCorpusView, predicate: Callable[[ParallelTextRow, int], bool]):
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class TakeParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(self, corpus: ParallelTextCorpusView, count: int):
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
