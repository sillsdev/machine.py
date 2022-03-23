from abc import abstractmethod
from typing import Generator, Iterable, Optional

from .text import Text
from .text_corpus_view import TextCorpusView
from .text_row import TextRow


class TextCorpus(TextCorpusView):
    @property
    @abstractmethod
    def texts(self) -> Iterable[Text]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> Optional[Text]:
        ...

    def get_text(self, id: str) -> Optional[Text]:
        return self[id]

    def _get_rows(self) -> Generator[TextRow, None, None]:
        for text in self.texts:
            with text.get_rows() as rows:
                yield from rows
