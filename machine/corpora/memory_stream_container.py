from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from .stream_container import StreamContainer


class MemoryStreamContainer(StreamContainer):
    def __init__(self, usfm: str) -> None:
        self._usfm = usfm

    def __enter__(self) -> MemoryStreamContainer:
        return self

    def __exit__(self, type, value, traceback) -> None: ...

    def open_stream(self) -> BinaryIO:
        return BytesIO(self._usfm.encode("utf-8"))

    def close(self) -> None: ...
