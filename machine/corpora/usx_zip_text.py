from typing import Optional

from ..scripture.verse_ref import Versification
from ..utils.typeshed import StrPath
from .stream_container import StreamContainer
from .usx_text_base import UsxTextBase
from .zip_entry_stream_container import ZipEntryStreamContainer


class UsxZipText(UsxTextBase):
    def __init__(
        self,
        id: str,
        archive_filename: StrPath,
        path: str,
        versification: Optional[Versification] = None,
    ) -> None:
        super().__init__(id, versification)

        self._archive_filename = archive_filename
        self._path = path

    def _create_stream_container(self) -> StreamContainer:
        return ZipEntryStreamContainer(self._archive_filename, self._path)
