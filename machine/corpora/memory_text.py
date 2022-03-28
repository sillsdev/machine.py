from typing import Generator, Iterable

from .corpora_utils import gen
from .text import Text
from .text_row import TextRow


class MemoryText(Text):
    def __init__(self, id: str, rows: Iterable[TextRow] = []) -> None:
        self._id = id
        self._rows = list(rows)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def _get_rows(self) -> Generator[TextRow, None, None]:
        return gen(self._rows)
