from pathlib import Path
from typing import Optional

from ..scripture.verse_ref import Versification
from ..tokenization import Tokenizer
from ..utils.typeshed import StrPath
from .corpora_helpers import get_usx_id
from .file_stream_container import FileStreamContainer
from .stream_container import StreamContainer
from .usx_text_base import UsxTextBase


class UsxFileText(UsxTextBase):
    def __init__(
        self, word_tokenizer: Tokenizer[str, int, str], filename: StrPath, versification: Optional[Versification] = None
    ) -> None:
        self._filename = Path(filename)
        super().__init__(word_tokenizer, get_usx_id(self._filename), versification)

    def _create_stream_container(self) -> StreamContainer:
        return FileStreamContainer(self._filename)
