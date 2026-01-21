from typing import Generator, Iterable

from .corpora_utils import gen
from .data_type import DataType
from .text import Text
from .text_row import TextRow


class MemoryText(Text):
    def __init__(self, id: str, rows: Iterable[TextRow] = [], data_type: DataType = DataType.SENTENCE) -> None:
        self._id = id
        self._rows = list(rows)
        if any([r.data_type != data_type for r in self._rows]):
            raise ValueError(f"{type(data_type)} of rows must match text {type(data_type)} {data_type}")
        self._data_type = data_type

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    @property
    def data_type(self) -> DataType:
        return self._data_type

    def _get_rows(self) -> Generator[TextRow, None, None]:
        return gen(self._rows)
