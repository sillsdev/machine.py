from io import TextIOWrapper
from typing import Generator, Optional

from ..scripture.verse_ref import VerseRef, are_overlapping_verse_ranges
from ..scripture.versification import Versification
from ..string_utils import has_sentence_ending
from ..tokenization.tokenizer import Tokenizer
from .corpora_helpers import merge_verse_ranges
from .scripture_text import ScriptureText
from .text_segment import TextSegment
from .usfm_marker import UsfmMarker
from .usfm_parser import UsfmParser
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmToken, UsfmTokenType


class UsfmTextBase(ScriptureText):
    def __init__(
        self,
        word_tokenizer: Tokenizer[str, int, str],
        id: str,
        stylesheet: UsfmStylesheet,
        encoding: str,
        versification: Optional[Versification],
        include_markers: bool,
    ) -> None:
        super().__init__(word_tokenizer, id, versification)

        self._parser = UsfmParser(stylesheet)
        self._encoding = encoding
        self._include_markers = include_markers

    def get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
        usfm = self._read_usfm()
        cur_embed_marker: Optional[UsfmMarker] = None
        in_wordlist_marker = False
        text = ""
        chapter: Optional[str] = None
        verse: Optional[str] = None
        sentence_start = True
        prev_token: Optional[UsfmToken] = None
        prev_verse_ref = VerseRef()

        for token in self._parser.parse(usfm):
            if token.type == UsfmTokenType.CHAPTER:
                if chapter is not None and verse is not None:
                    for seg in self._create_text_segments(
                        include_text, prev_verse_ref, chapter, verse, text, sentence_start
                    ):
                        yield seg
                    sentence_start = True
                    text = ""
                chapter = token.text
                verse = None
            elif token.type == UsfmTokenType.VERSE:
                assert token.text is not None
                if chapter is not None and verse is not None:
                    if token.text == verse:
                        for seg in self._create_text_segments(
                            include_text, prev_verse_ref, chapter, verse, text, sentence_start
                        ):
                            yield seg
                        sentence_start = has_sentence_ending(text)
                        text = ""

                        # ignore duplicate verse
                        verse = None
                    elif are_overlapping_verse_ranges(token.text, verse):
                        verse = merge_verse_ranges(token.text, verse)
                    else:
                        for seg in self._create_text_segments(
                            include_text, prev_verse_ref, chapter, verse, text, sentence_start
                        ):
                            yield seg
                        sentence_start = has_sentence_ending(text)
                        text = ""
                        verse = token.text
                else:
                    verse = token.text

    def _read_usfm(self) -> str:
        with self._create_stream_container() as stream_container, TextIOWrapper(
            stream_container.open_stream(), encoding=self._encoding
        ) as reader:
            return reader.read()
