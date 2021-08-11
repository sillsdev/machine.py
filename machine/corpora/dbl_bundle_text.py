from typing import Optional

from ..scripture.verse_ref import Versification
from ..tokenization.tokenizer import Tokenizer
from ..utils.typeshed import StrPath
from .stream_container import StreamContainer
from .usx_text_base import UsxTextBase
from .zip_entry_stream_container import ZipEntryStreamContainer


class DblBundleText(UsxTextBase):
    def __init__(
        self,
        word_tokenizer: Tokenizer[str, int, str],
        id: str,
        bundle_filename: StrPath,
        path: str,
        versification: Optional[Versification] = None,
    ) -> None:
        super().__init__(word_tokenizer, id, versification)

        self._bundle_filename = bundle_filename
        self._path = path

    def _create_stream_container(self) -> StreamContainer:
        return ZipEntryStreamContainer(self._bundle_filename, self._path)
