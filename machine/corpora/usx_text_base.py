from abc import abstractmethod
from typing import Generator, Optional

from ..scripture.verse_ref import Versification
from ..tokenization.tokenizer import Tokenizer
from .scripture_text import ScriptureText
from .stream_container import StreamContainer
from .text_segment import TextSegment
from .usx_verse_parser import UsxVerseParser


class UsxTextBase(ScriptureText):
    def __init__(
        self, word_tokenizer: Tokenizer[str, int, str], id: str, versification: Optional[Versification]
    ) -> None:
        super().__init__(word_tokenizer, id, versification)
        self._parser = UsxVerseParser()

    @abstractmethod
    def _create_stream_container(self) -> StreamContainer:
        ...

    def _get_segments_in_doc_order(self, include_text: bool) -> Generator[TextSegment, None, None]:
        with self._create_stream_container() as stream_container, stream_container.open_stream() as stream:
            for verse in self._parser.parse(stream):
                yield from self._create_text_segments(
                    include_text, verse.chapter, verse.verse, verse.text, verse.is_sentence_start
                )
