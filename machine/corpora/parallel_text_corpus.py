from itertools import islice
from typing import Any, Callable, Generator, Iterable, Optional, Sequence, Tuple

from ..tokenization.tokenizer import Tokenizer
from .corpora_utils import get_split_indices
from .corpus import Corpus
from .parallel_text_row import ParallelTextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class ParallelTextCorpus(Corpus[ParallelTextRow]):
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

    def transform(self, transform: Callable[[ParallelTextRow], ParallelTextRow]) -> "ParallelTextCorpus":
        return _TransformParallelTextCorpus(self, transform)

    def filter_nonempty(self) -> "ParallelTextCorpus":
        return self.filter(lambda r: not r.is_empty)

    def filter(self, predicate: Callable[[ParallelTextRow], bool]) -> "ParallelTextCorpus":
        return self.filter_by_index(lambda r, _: predicate(r))

    def filter_by_index(self, predicate: Callable[[ParallelTextRow, int], bool]) -> "ParallelTextCorpus":
        return _FilterParallelTextCorpus(self, predicate)

    def take(self, count: int) -> "ParallelTextCorpus":
        return _TakeParallelTextCorpus(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple["ParallelTextCorpus", "ParallelTextCorpus", int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda r, i: i not in split_indices and (include_empty or not r.is_empty))
        split_corpus = self.filter_by_index(lambda r, i: i in split_indices and (include_empty or not r.is_empty))

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def to_tuples(self) -> Iterable[Tuple[Sequence[str], Sequence[str]]]:
        return self.map(lambda r: (r.source_segment, r.target_segment))


class _TransformParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, transform: Callable[[ParallelTextRow], ParallelTextRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def missing_rows_allowed(self) -> bool:
        return self._corpus.missing_rows_allowed

    def count(self, include_empty: bool = True) -> int:
        return self._corpus.count(include_empty)

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class _FilterParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, predicate: Callable[[ParallelTextRow, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
