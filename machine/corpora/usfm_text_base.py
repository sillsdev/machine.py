from io import TextIOWrapper
from typing import Generator, Optional

from ..scripture.verse_ref import VerseRef, Versification, are_overlapping_verse_ranges
from ..tokenization.tokenizer import Tokenizer
from ..utils.string_utils import has_sentence_ending, is_integer
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

    def _get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
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
            elif token.type == UsfmTokenType.PARAGRAPH:
                if chapter is not None and verse is not None and not _is_verse_para(token):
                    for seg in self._create_text_segments(
                        include_text, prev_verse_ref, chapter, verse, text, sentence_start
                    ):
                        yield seg
                    sentence_start = True
                    text = ""
                    verse = None
            elif token.type == UsfmTokenType.NOTE:
                cur_embed_marker = token.marker
                if chapter is not None and verse is not None and self._include_markers:
                    if (
                        prev_token is not None
                        and prev_token.type == UsfmTokenType.PARAGRAPH
                        and _is_verse_para(prev_token)
                    ):
                        text += str(prev_token) + " "
                    text += str(token) + " "
            elif token.type == UsfmTokenType.END:
                assert token.marker is not None
                if cur_embed_marker is not None and token.marker.marker == cur_embed_marker.end_marker:
                    cur_embed_marker = None
                if in_wordlist_marker and token.marker.marker == "w*":
                    in_wordlist_marker = False
                if chapter is not None and verse is not None and self._include_markers:
                    text += str(token)
            elif token.type == UsfmTokenType.CHARACTER:
                assert token.marker is not None
                if token.marker.marker == "fig" or token.marker.marker == "va" or token.marker.marker == "vp":
                    cur_embed_marker = token.marker
                elif token.marker.marker == "w":
                    in_wordlist_marker = True
                if chapter is not None and verse is not None and self._include_markers:
                    if (
                        prev_token is not None
                        and prev_token.type == UsfmTokenType.PARAGRAPH
                        and _is_verse_para(prev_token)
                    ):
                        text += str(prev_token) + " "
                    text += str(token) + " "
            elif token.type == UsfmTokenType.TEXT:
                if chapter is not None and verse is not None and token.text is not None and token.text != "":
                    if self._include_markers:
                        if (
                            prev_token is not None
                            and prev_token.type == UsfmTokenType.PARAGRAPH
                            and _is_verse_para(prev_token)
                        ):
                            text += str(prev_token)
                            text += " "
                        text += str(token)
                    elif cur_embed_marker is None:
                        token_text = token.text
                        if in_wordlist_marker:
                            index = token_text.find("|")
                            if index >= 0:
                                token_text = token_text[:index]

                        if (
                            prev_token is not None
                            and prev_token.type == UsfmTokenType.END
                            and (text == "" or text[-1].isspace())
                        ):
                            token_text = token_text.lstrip()
                        text += token_text
            prev_token = token

        if chapter is not None and verse is not None:
            for seg in self._create_text_segments(include_text, prev_verse_ref, chapter, verse, text, sentence_start):
                yield seg

    def _read_usfm(self) -> str:
        with self._create_stream_container() as stream_container, TextIOWrapper(
            stream_container.open_stream(), encoding=self._encoding
        ) as reader:
            return reader.read()


_NONVERSE_PARA_STYLES = {"ms", "mr", "s", "sr", "r", "d", "sp", "rem", "restore"}


def _is_numbered_style(style_prefix: str, style: str) -> bool:
    return style.startswith(style_prefix) and is_integer(style[len(style_prefix) :])


def _is_verse_para(para_token: UsfmToken) -> bool:
    assert para_token.marker is not None
    style = para_token.marker.marker
    if style in _NONVERSE_PARA_STYLES:
        return False

    if _is_numbered_style("ms", style):
        return False

    if _is_numbered_style("s", style):
        return False

    return True
