from itertools import islice
from random import Random
from typing import Any, Callable, Generator, Optional, Tuple

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
        return FilterTextCorpusView(self, lambda row, _: predicate(row))

    def filter_by_index(self, predicate: Callable[[TextRow, int], bool]) -> "TextCorpusView":
        return FilterTextCorpusView(self, predicate)

    def transform(self, transform: Callable[[TextRow], TextRow]) -> "TextCorpusView":
        return TransformTextCorpusView(self, transform)

    def take(self, count: int) -> "TextCorpusView":
        return TakeTextCorpusView(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, seed: Any = None
    ) -> Tuple["TextCorpusView", "TextCorpusView", int, int]:
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
    def __init__(self, corpus: TextCorpusView, predicate: Callable[[TextRow, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class TakeTextCorpusView(TextCorpusView):
    def __init__(self, corpus: TextCorpusView, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
