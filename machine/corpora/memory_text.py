from typing import Generator, Iterable

from .corpora_helpers import gen
from .text import Text
from .text_corpus_row import TextCorpusRow


class MemoryText(Text):
    def __init__(self, id: str, rows: Iterable[TextCorpusRow] = []) -> None:
        self._id = id
        self._rows = list(rows)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def _get_rows(self) -> Generator[TextCorpusRow, None, None]:
        return gen(self._rows)
