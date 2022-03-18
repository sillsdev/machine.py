from abc import ABC, abstractmethod
from typing import Callable, Generator, Optional

from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .parallel_text_corpus_row import ParallelTextCorpusRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class ParallelTextCorpusView(ABC):
    @property
    @abstractmethod
    def source(self) -> "ParallelTextCorpusView":
        ...

    def get_rows(
        self, all_source_rows: bool = False, all_target_rows: bool = False
    ) -> ContextManagedGenerator[ParallelTextCorpusRow, None, None]:
        return ContextManagedGenerator(self._get_rows(all_source_rows, all_target_rows))

    @abstractmethod
    def _get_rows(self, all_source_rows: bool, all_target_rows: bool) -> Generator[ParallelTextCorpusRow, None, None]:
        ...

    def get_count(self, all_source_rows: bool = False, all_target_rows: bool = False) -> int:
        with self.get_rows(all_source_rows, all_target_rows) as rows:
            return sum(1 for _ in rows)

    def invert(self) -> "ParallelTextCorpusView":
        def _invert(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            return row.invert()

        return TransformParallelTextCorpusView(self, _invert)

    def tokenize(
        self, source_tokenizer: Tokenizer[str, int, str], target_tokenizer: Optional[Tokenizer[str, int, str]] = None
    ) -> "ParallelTextCorpusView":
        if target_tokenizer is None:
            target_tokenizer = source_tokenizer

        def _tokenize(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = list(source_tokenizer.tokenize(row.source_text))
            row.target_segment = list(target_tokenizer.tokenize(row.target_text))
            return row

        return TransformParallelTextCorpusView(self, _tokenize)

    def detokenize(
        self, source_detokenizer: Detokenizer[str, str], target_detokenizer: Optional[Detokenizer[str, str]] = None
    ) -> "ParallelTextCorpusView":
        if target_detokenizer is None:
            target_detokenizer = source_detokenizer

        def _detokenize(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = [source_detokenizer.detokenize(row.source_segment)]
            row.target_segment = [target_detokenizer.detokenize(row.target_segment)]
            return row

        return TransformParallelTextCorpusView(self, _detokenize)

    def normalize(self, normalization_form: str) -> "ParallelTextCorpusView":
        def _normalize(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = normalize(normalization_form, row.source_segment)
            row.target_segment = normalize(normalization_form, row.target_segment)
            return row

        return TransformParallelTextCorpusView(self, _normalize)

    def nfc_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFC")

    def nfd_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFD")

    def nfkc_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> "ParallelTextCorpusView":
        return self.normalize("NFKD")

    def lowercase(self) -> "ParallelTextCorpusView":
        def _lowercase(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = lowercase(row.source_segment)
            row.target_segment = lowercase(row.target_segment)
            return row

        return TransformParallelTextCorpusView(self, _lowercase)

    def escape_spaces(self) -> "ParallelTextCorpusView":
        def _escape_spaces(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = escape_spaces(row.source_segment)
            row.target_segment = escape_spaces(row.target_segment)
            return row

        return TransformParallelTextCorpusView(self, _escape_spaces)

    def unescape_spaces(self) -> "ParallelTextCorpusView":
        def _unescape_spaces(row: ParallelTextCorpusRow) -> ParallelTextCorpusRow:
            row.source_segment = unescape_spaces(row.source_segment)
            row.target_segment = unescape_spaces(row.target_segment)
            return row

        return TransformParallelTextCorpusView(self, _unescape_spaces)

    def filter(self, predicate: Callable[[ParallelTextCorpusRow], bool]) -> "ParallelTextCorpusView":
        return FilterParallelTextCorpusView(self, predicate)


class TransformParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(
        self, corpus: ParallelTextCorpusView, transform: Callable[[ParallelTextCorpusRow], ParallelTextCorpusRow]
    ):
        self._corpus = corpus
        self._transform = transform

    @property
    def source(self) -> ParallelTextCorpusView:
        return self._corpus.source

    def _get_rows(self, all_source_rows: bool, all_target_rows: bool) -> Generator[ParallelTextCorpusRow, None, None]:
        with self._corpus.get_rows(all_source_rows, all_target_rows) as rows:
            for row in rows:
                yield self._transform(row)


class FilterParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(self, corpus: ParallelTextCorpusView, predicate: Callable[[ParallelTextCorpusRow], bool]):
        self._corpus = corpus
        self._predicate = predicate

    @property
    def source(self) -> ParallelTextCorpusView:
        return self._corpus.source

    def _get_rows(self, all_source_rows: bool, all_target_rows: bool) -> Generator[ParallelTextCorpusRow, None, None]:
        with self._corpus.get_rows(all_source_rows, all_target_rows) as rows:
            for row in rows:
                if self._predicate(row):
                    yield row
