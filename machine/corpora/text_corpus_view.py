from abc import ABC, abstractmethod
from typing import Callable, Generator, Optional

from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .text_corpus_row import TextCorpusRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class TextCorpusView(ABC):
    @property
    @abstractmethod
    def source(self) -> "TextCorpusView":
        ...

    def get_rows(
        self, based_on: Optional["TextCorpusView"] = None
    ) -> ContextManagedGenerator[TextCorpusRow, None, None]:
        return ContextManagedGenerator(self._get_rows(based_on))

    @abstractmethod
    def _get_rows(self, based_on: Optional["TextCorpusView"]) -> Generator[TextCorpusRow, None, None]:
        ...

    def get_count(self) -> int:
        with self.get_rows() as rows:
            return sum(1 for _ in rows)

    def tokenize(self, tokenizer: Tokenizer[str, int, str]) -> "TextCorpusView":
        def _tokenize(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = list(tokenizer.tokenize(row.text))
            return row

        return TransformTextCorpusView(self, _tokenize)

    def detokenize(self, detokenizer: Detokenizer[str, str]) -> "TextCorpusView":
        def _detokenize(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = [detokenizer.detokenize(row.segment)]
            return row

        return TransformTextCorpusView(self, _detokenize)

    def normalize(self, normalization_form: str) -> "TextCorpusView":
        def _normalize(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = normalize(normalization_form, row.segment)
            return row

        return TransformTextCorpusView(self, _normalize)

    def nfc_normalize(self) -> "TextCorpusView":
        return self.normalize("NFC")

    def nfd_normalize(self) -> "TextCorpusView":
        return self.normalize("NFD")

    def nfkc_normalize(self) -> "TextCorpusView":
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> "TextCorpusView":
        return self.normalize("NFKD")

    def lowercase(self) -> "TextCorpusView":
        def _lowercase(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = lowercase(row.segment)
            return row

        return TransformTextCorpusView(self, _lowercase)

    def escape_spaces(self) -> "TextCorpusView":
        def _escape_spaces(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = escape_spaces(row.segment)
            return row

        return TransformTextCorpusView(self, _escape_spaces)

    def unescape_spaces(self) -> "TextCorpusView":
        def _unescape_spaces(row: TextCorpusRow) -> TextCorpusRow:
            row.segment = unescape_spaces(row.segment)
            return row

        return TransformTextCorpusView(self, _unescape_spaces)

    def filter(self, predicate: Callable[[TextCorpusRow], bool]) -> "TextCorpusView":
        return FilterTextCorpusView(self, predicate)


class TransformTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, transform: Callable[[TextCorpusRow], TextCorpusRow]) -> None:
        self._corpus = corpus
        self._transform = transform

    @property
    def source(self) -> TextCorpusView:
        return self._corpus.source

    def _get_rows(self, based_on: Optional[TextCorpusView] = None) -> Generator[TextCorpusRow, None, None]:
        with self._corpus.get_rows(based_on) as rows:
            for row in rows:
                yield self._transform(row)


class FilterTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, predicate: Callable[[TextCorpusRow], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    @property
    def source(self) -> TextCorpusView:
        return self._corpus.source

    def _get_rows(self, based_on: Optional[TextCorpusView] = None) -> Generator[TextCorpusRow, None, None]:
        with self._corpus.get_rows(based_on) as rows:
            for row in rows:
                if self._predicate(row):
                    yield row
