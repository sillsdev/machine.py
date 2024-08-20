from pathlib import Path
from typing import Optional

from ..scripture.verse_ref import Versification
from ..utils.typeshed import StrPath
from .file_stream_container import FileStreamContainer
from .stream_container import StreamContainer
from .usfm_stylesheet import UsfmStylesheet
from .usfm_text_base import UsfmTextBase


class UsfmFileText(UsfmTextBase):
    def __init__(
        self,
        stylesheet: UsfmStylesheet,
        encoding: str,
        id: str,
        filename: StrPath,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
        include_all_text: bool = False,
        project: Optional[str] = None,
    ) -> None:
        super().__init__(id, stylesheet, encoding, versification, include_markers, include_all_text, project)

        self._filename = Path(filename)

    def _create_stream_container(self) -> StreamContainer:
        return FileStreamContainer(self._filename)
