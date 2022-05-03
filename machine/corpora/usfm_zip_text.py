from io import TextIOWrapper
from typing import Optional
from zipfile import ZipFile

from ..scripture.verse_ref import Versification
from ..utils.typeshed import StrPath
from .stream_container import StreamContainer
from .usfm_stylesheet import UsfmStylesheet
from .usfm_text_base import UsfmTextBase
from .zip_entry_stream_container import ZipEntryStreamContainer


class UsfmZipText(UsfmTextBase):
    def __init__(
        self,
        stylesheet: UsfmStylesheet,
        encoding: str,
        archive_filename: StrPath,
        path: str,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
    ) -> None:
        super().__init__(
            _get_id(archive_filename, path, encoding), stylesheet, encoding, versification, include_markers
        )
        self._archive_filename = archive_filename
        self._path = path

    def _create_stream_container(self) -> StreamContainer:
        return ZipEntryStreamContainer(self._archive_filename, self._path)


def _get_id(archive_filename: StrPath, path: str, encoding: str) -> str:
    with ZipFile(archive_filename, "r") as archive:
        entry = next((zi for zi in archive.filelist if zi.filename == path))
        with archive.open(entry, "r") as file:
            stream = TextIOWrapper(file, encoding=encoding)
            for line in stream:
                line = line.strip()
                if line.startswith("\\id "):
                    id = line[4:]
                    index = id.find(" ")
                    if index != -1:
                        id = id[:index]
                    return id.strip()
    raise RuntimeError("The USFM does not contain and 'id' marker.")
