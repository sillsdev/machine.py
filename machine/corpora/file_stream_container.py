from typing import Any, BinaryIO

from ..utils.typeshed import StrPath
from .stream_container import StreamContainer


class FileStreamContainer(StreamContainer):
    def __init__(self, filename: StrPath) -> None:
        self._filename = filename

    def __enter__(self) -> "FileStreamContainer":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        ...

    def open_stream(self) -> BinaryIO:
        return open(self._filename, "rb")

    def close(self) -> None:
        ...
