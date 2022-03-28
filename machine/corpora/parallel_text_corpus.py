from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Callable, Generator, Iterable, Optional, Tuple

from ..tokenization.tokenizer import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_utils import get_split_indices
from .parallel_text_row import ParallelTextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class ParallelTextCorpus(ABC, Iterable[ParallelTextRow]):
    def get_rows(self) -> ContextManagedGenerator[ParallelTextRow, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        ...

    def __iter__(self) -> ContextManagedGenerator[ParallelTextRow, None, None]:
        return self.get_rows()

    def count(self) -> int:
        with self.get_rows() as rows:
            return sum(1 for _ in rows)

    def invert(self) -> "ParallelTextCorpus":
        def _invert(row: ParallelTextRow) -> ParallelTextRow:
            return row.invert()

        return self.transform(_invert)

    def tokenize(
        self, source_tokenizer: Tokenizer[str, int, str], target_tokenizer: Optional[Tokenizer[str, int, str]] = None
    ) -> "ParallelTextCorpus":
        if target_tokenizer is None:
            target_tokenizer = source_tokenizer

        def _tokenize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = list(source_tokenizer.tokenize(row.source_text))
            row.target_segment = list(target_tokenizer.tokenize(row.target_text))
            return row

        return self.transform(_tokenize)

    def normalize(self, normalization_form: str) -> "ParallelTextCorpus":
        def _normalize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = normalize(normalization_form, row.source_segment)
            row.target_segment = normalize(normalization_form, row.target_segment)
            return row

        return self.transform(_normalize)

    def nfc_normalize(self) -> "ParallelTextCorpus":
        return self.normalize("NFC")

    def nfd_normalize(self) -> "ParallelTextCorpus":
        return self.normalize("NFD")

    def nfkc_normalize(self) -> "ParallelTextCorpus":
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> "ParallelTextCorpus":
        return self.normalize("NFKD")

    def lowercase(self) -> "ParallelTextCorpus":
        def _lowercase(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = lowercase(row.source_segment)
            row.target_segment = lowercase(row.target_segment)
            return row

        return self.transform(_lowercase)

    def escape_spaces(self) -> "ParallelTextCorpus":
        def _escape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = escape_spaces(row.source_segment)
            row.target_segment = escape_spaces(row.target_segment)
            return row

        return self.transform(_escape_spaces)

    def unescape_spaces(self) -> "ParallelTextCorpus":
        def _unescape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = unescape_spaces(row.source_segment)
            row.target_segment = unescape_spaces(row.target_segment)
            return row

        return self.transform(_unescape_spaces)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, seed: Any = None
    ) -> Tuple["ParallelTextCorpus", "ParallelTextCorpus", int, int]:
        corpus_size = self.count()
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda _, i: i not in split_indices)
        split_corpus = self.filter_by_index(lambda _, i: i in split_indices)

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def filter(self, predicate: Callable[[ParallelTextRow], bool]) -> "ParallelTextCorpus":
        return _FilterParallelTextCorpus(self, lambda row, _: predicate(row))

    def filter_by_index(self, predicate: Callable[[ParallelTextRow, int], bool]) -> "ParallelTextCorpus":
        return _FilterParallelTextCorpus(self, predicate)

    def transform(self, transform: Callable[[ParallelTextRow], ParallelTextRow]) -> "ParallelTextCorpus":
        return _TransformParallelTextCorpus(self, transform)

    def take(self, count: int) -> "ParallelTextCorpus":
        return _TakeParallelTextCorpus(self, count)


class _TransformParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, transform: Callable[[ParallelTextRow], ParallelTextRow]):
        self._corpus = corpus
        self._transform = transform

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class _FilterParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, predicate: Callable[[ParallelTextRow, int], bool]):
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, count: int):
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
