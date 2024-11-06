from typing import Optional

from ..scripture.verse_ref import Versification
from .memory_stream_container import MemoryStreamContainer
from .stream_container import StreamContainer
from .usx_text_base import UsxTextBase


class UsxMemoryText(UsxTextBase):
    def __init__(self, id: str, usx: str, versification: Optional[Versification] = None) -> None:
        super().__init__(id, versification)
        self._usx = usx

    def _create_stream_container(self) -> StreamContainer:
        return MemoryStreamContainer(self._usx)
