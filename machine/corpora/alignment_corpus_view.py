from itertools import islice
from typing import Callable, Generator

from .alignment_row import AlignmentRow
from .corpus_view import CorpusView


class AlignmentCorpusView(CorpusView[AlignmentRow]):
    def invert(self) -> "AlignmentCorpusView":
        def _invert(row: AlignmentRow) -> AlignmentRow:
            return row.invert()

        return self.transform(_invert)

    def filter(self, predicate: Callable[[AlignmentRow], bool]) -> "AlignmentCorpusView":
        return FilterAlignmentCorpusView(self, predicate)

    def transform(self, transform: Callable[[AlignmentRow], AlignmentRow]) -> "AlignmentCorpusView":
        return TransformAlignmentCorpusView(self, transform)

    def take(self, count: int) -> "AlignmentCorpusView":
        return TakeAlignmentCorpusView(self, count)


class TransformAlignmentCorpusView(AlignmentCorpusView):
    def __init__(self, corpus: AlignmentCorpusView, transform: Callable[[AlignmentRow], AlignmentRow]):
        self._corpus = corpus
        self._transform = transform

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class FilterAlignmentCorpusView(AlignmentCorpusView):
    def __init__(self, corpus: AlignmentCorpusView, predicate: Callable[[AlignmentRow], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from filter(self._predicate, rows)


class TakeAlignmentCorpusView(AlignmentCorpusView):
    def __init__(self, corpus: AlignmentCorpusView, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)
