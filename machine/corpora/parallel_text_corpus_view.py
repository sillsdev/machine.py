from itertools import islice
from typing import Callable, Generator, Optional

from ..tokenization.detokenizer import Detokenizer
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

    def detokenize(
        self, source_detokenizer: Detokenizer[str, str], target_detokenizer: Optional[Detokenizer[str, str]] = None
    ) -> "ParallelTextCorpusView":
        if target_detokenizer is None:
            target_detokenizer = source_detokenizer

        def _detokenize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = [source_detokenizer.detokenize(row.source_segment)]
            row.target_segment = [target_detokenizer.detokenize(row.target_segment)]
            return row

        return self.transform(_detokenize)

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

    def filter(self, predicate: Callable[[ParallelTextRow], bool]) -> "ParallelTextCorpusView":
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
    def __init__(self, corpus: ParallelTextCorpusView, predicate: Callable[[ParallelTextRow], bool]):
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from filter(self._predicate, rows)


class TakeParallelTextCorpusView(ParallelTextCorpusView):
    def __init__(self, corpus: ParallelTextCorpusView, count: int):
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
