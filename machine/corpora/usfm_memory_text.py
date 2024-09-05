from typing import Optional

from ..scripture.verse_ref import Versification
from .memory_stream_container import MemoryStreamContainer
from .stream_container import StreamContainer
from .usfm_stylesheet import UsfmStylesheet
from .usfm_text_base import UsfmTextBase


class UsfmMemoryText(UsfmTextBase):
    def __init__(
        self,
        stylesheet: UsfmStylesheet,
        encoding: str,
        id: str,
        usfm: str,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
        include_all_text: bool = False,
    ) -> None:
        super().__init__(id, stylesheet, encoding, versification, include_markers, include_all_text)

        self._usfm = usfm

    def _create_stream_container(self) -> StreamContainer:
        return MemoryStreamContainer(self._usfm)
