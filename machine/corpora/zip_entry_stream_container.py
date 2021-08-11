from io import BytesIO
from typing import Any, BinaryIO
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .stream_container import StreamContainer


class ZipEntryStreamContainer(StreamContainer):
    def __init__(self, archive_filename: StrPath, entry_path: str) -> None:
        self._archive = ZipFile(archive_filename, "r")
        self._entry = self._archive.getinfo(entry_path)

    def __enter__(self) -> "ZipEntryStreamContainer":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def open_stream(self) -> BinaryIO:
        return BytesIO(self._archive.read(self._entry))

    def close(self) -> None:
        self._archive.close()
