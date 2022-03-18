from abc import abstractmethod
from typing import Generator, Iterable, Optional

from .text import Text
from .text_corpus_row import TextCorpusRow
from .text_corpus_view import TextCorpusView


class TextCorpus(TextCorpusView):
    @property
    @abstractmethod
    def texts(self) -> Iterable[Text]:
        ...

    @property
    def source(self) -> TextCorpusView:
        return self

    @abstractmethod
    def __getitem__(self, id: str) -> Optional[Text]:
        ...

    def get_text(self, id: str) -> Optional[Text]:
        return self[id]

    def _get_rows(self, based_on: Optional[TextCorpusView]) -> Generator[TextCorpusRow, None, None]:
        for text in self.texts:
            with text.get_rows() as rows:
                yield from rows
