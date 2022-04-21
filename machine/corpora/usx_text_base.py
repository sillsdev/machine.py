from abc import abstractmethod
from typing import Generator, Optional

from ..scripture.verse_ref import Versification
from .scripture_text import ScriptureText
from .stream_container import StreamContainer
from .text_row import TextRow
from .usx_verse_parser import UsxVerseParser


class UsxTextBase(ScriptureText):
    def __init__(self, id: str, versification: Optional[Versification]) -> None:
        super().__init__(id, versification)
        self._parser = UsxVerseParser()

    @abstractmethod
    def _create_stream_container(self) -> StreamContainer:
        ...

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with self._create_stream_container() as stream_container, stream_container.open_stream() as stream:
            for verse in self._parser.parse(stream):
                verse_ref = self._create_verse_ref(verse.chapter, verse.verse)
                yield from self._create_rows(verse_ref, verse.text, verse.is_sentence_start)
