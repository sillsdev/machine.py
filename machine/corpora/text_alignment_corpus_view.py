from abc import ABC, abstractmethod
from typing import Callable, Generator

from ..utils.context_managed_generator import ContextManagedGenerator
from .text_alignment_corpus_row import TextAlignmentCorpusRow


class TextAlignmentCorpusView(ABC):
    @property
    @abstractmethod
    def source(self) -> "TextAlignmentCorpusView":
        ...

    def get_rows(self) -> ContextManagedGenerator[TextAlignmentCorpusRow, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[TextAlignmentCorpusRow, None, None]:
        ...

    def get_count(self) -> int:
        with self.get_rows() as rows:
            return sum(1 for _ in rows)

    def invert(self) -> "TextAlignmentCorpusView":
        def _invert(row: TextAlignmentCorpusRow) -> TextAlignmentCorpusRow:
            return row.invert()

        return TransformTextAlignmentCorpusView(self, _invert)


class TransformTextAlignmentCorpusView(TextAlignmentCorpusView):
    def __init__(
        self, corpus: TextAlignmentCorpusView, transform: Callable[[TextAlignmentCorpusRow], TextAlignmentCorpusRow]
    ):
        self._corpus = corpus
        self._transform = transform

    @property
    def source(self) -> TextAlignmentCorpusView:
        return self._corpus.source

    def _get_rows(self) -> Generator[TextAlignmentCorpusRow, None, None]:
        with self._corpus.get_rows() as rows:
            for row in rows:
                yield self._transform(row)
