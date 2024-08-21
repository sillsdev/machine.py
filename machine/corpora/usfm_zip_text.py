from typing import Optional

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
        id: str,
        archive_filename: StrPath,
        path: str,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
        include_all_text: bool = False,
        project: Optional[str] = None,
    ) -> None:
        super().__init__(id, stylesheet, encoding, versification, include_markers, include_all_text, project)
        self._archive_filename = archive_filename
        self._path = path

    def _create_stream_container(self) -> StreamContainer:
        return ZipEntryStreamContainer(self._archive_filename, self._path)
