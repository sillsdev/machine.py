from itertools import islice
from typing import Callable, Generator, Optional

from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from .alignment_corpus_view import AlignmentCorpusView
from .corpus_view import CorpusView
from .parallel_text_corpus_view import ParallelTextCorpusView
from .text_row import TextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class TextCorpusView(CorpusView[TextRow]):
    def tokenize(self, tokenizer: Tokenizer[str, int, str]) -> "TextCorpusView":
        def _tokenize(row: TextRow) -> TextRow:
            row.segment = list(tokenizer.tokenize(row.text))
            return row

        return self.transform(_tokenize)

    def detokenize(self, detokenizer: Detokenizer[str, str]) -> "TextCorpusView":
        def _detokenize(row: TextRow) -> TextRow:
            row.segment = [detokenizer.detokenize(row.segment)]
            return row

        return self.transform(_detokenize)

    def normalize(self, normalization_form: str) -> "TextCorpusView":
        def _normalize(row: TextRow) -> TextRow:
            row.segment = normalize(normalization_form, row.segment)
            return row

        return self.transform(_normalize)

    def nfc_normalize(self) -> "TextCorpusView":
        return self.normalize("NFC")

    def nfd_normalize(self) -> "TextCorpusView":
        return self.normalize("NFD")

    def nfkc_normalize(self) -> "TextCorpusView":
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> "TextCorpusView":
        return self.normalize("NFKD")

    def lowercase(self) -> "TextCorpusView":
        def _lowercase(row: TextRow) -> TextRow:
            row.segment = lowercase(row.segment)
            return row

        return self.transform(_lowercase)

    def escape_spaces(self) -> "TextCorpusView":
        def _escape_spaces(row: TextRow) -> TextRow:
            row.segment = escape_spaces(row.segment)
            return row

        return self.transform(_escape_spaces)

    def unescape_spaces(self) -> "TextCorpusView":
        def _unescape_spaces(row: TextRow) -> TextRow:
            row.segment = unescape_spaces(row.segment)
            return row

        return self.transform(_unescape_spaces)

    def filter(self, predicate: Callable[[TextRow], bool]) -> "TextCorpusView":
        return FilterTextCorpusView(self, predicate)

    def transform(self, transform: Callable[[TextRow], TextRow]) -> "TextCorpusView":
        return TransformTextCorpusView(self, transform)

    def take(self, count: int) -> "TextCorpusView":
        return TakeTextCorpusView(self, count)

    def align_rows(
        self,
        other: "TextCorpusView",
        alignment_corpus: Optional[AlignmentCorpusView] = None,
        all_source_rows: bool = False,
        all_target_rows: bool = False,
    ) -> ParallelTextCorpusView:
        from .parallel_text_corpus import ParallelTextCorpus

        return ParallelTextCorpus(self, other, alignment_corpus, all_source_rows, all_target_rows)


class TransformTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, transform: Callable[[TextRow], TextRow]) -> None:
        self._corpus = corpus
        self._transform = transform

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class FilterTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, predicate: Callable[[TextRow], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from filter(self._predicate, rows)


class TakeTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
