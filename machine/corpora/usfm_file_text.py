from pathlib import Path
from typing import Optional

from ..scripture.verse_ref import Versification
from ..tokenization import Tokenizer
from ..utils.typeshed import StrPath
from .file_stream_container import FileStreamContainer
from .stream_container import StreamContainer
from .usfm_stylesheet import UsfmStylesheet
from .usfm_text_base import UsfmTextBase


class UsfmFileText(UsfmTextBase):
    def __init__(
        self,
        word_tokenizer: Tokenizer[str, int, str],
        stylesheet: UsfmStylesheet,
        encoding: str,
        filename: StrPath,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
    ) -> None:
        super().__init__(
            word_tokenizer, _get_id(filename, encoding), stylesheet, encoding, versification, include_markers
        )

        self._filename = Path(filename)

    def _create_stream_container(self) -> StreamContainer:
        return FileStreamContainer(self._filename)


def _get_id(filename: StrPath, encoding: str) -> str:
    with open(filename, "r", encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line.startswith("\\id "):
                id = line[4:]
                index = id.find(" ")
                if index != -1:
                    id = id[:index]
                return id.strip()
    raise RuntimeError("The USFM does not contain and 'id' marker.")
