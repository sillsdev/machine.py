from abc import abstractmethod
from io import TextIOWrapper
from typing import Generator, Optional

from ..scripture.verse_ref import Versification, are_overlapping_verse_ranges
from ..utils.string_utils import has_sentence_ending, is_integer
from .corpora_utils import merge_verse_ranges
from .scripture_text import ScriptureText
from .stream_container import StreamContainer
from .text_row import TextRow
from .usfm_marker import UsfmMarker, UsfmTextType
from .usfm_parser import UsfmParser
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmToken, UsfmTokenType


class UsfmTextBase(ScriptureText):
    def __init__(
        self,
        id: str,
        stylesheet: UsfmStylesheet,
        encoding: str,
        versification: Optional[Versification],
        include_markers: bool,
    ) -> None:
        super().__init__(id, versification)

        self._parser = UsfmParser(stylesheet)
        self._encoding = encoding
        self._include_markers = include_markers

    @abstractmethod
    def _create_stream_container(self) -> StreamContainer:
        ...

    def _get_rows(self) -> Generator[TextRow, None, None]:
        usfm = self._read_usfm()
        cur_embed_marker: Optional[UsfmMarker] = None
        cur_span_marker: Optional[UsfmMarker] = None
        verse_text = ""
        chapter: Optional[str] = None
        verse: Optional[str] = None
        sentence_start = True
        prev_token: Optional[UsfmToken] = None
        is_verse_para = False
        for token in self._parser.parse(usfm, preserve_whitespace=self._include_markers):
            if token.type == UsfmTokenType.CHAPTER:
                if chapter is not None and verse is not None:
                    yield from self._create_rows(chapter, verse, verse_text, sentence_start)
                    sentence_start = True
                    verse_text = ""
                chapter = token.data
                verse = None
            elif token.type == UsfmTokenType.VERSE:
                assert token.data is not None
                if chapter is not None and verse is not None:
                    if token.data == verse:
                        yield from self._create_rows(chapter, verse, verse_text, sentence_start)
                        sentence_start = has_sentence_ending(verse_text)
                        verse_text = ""

                        # ignore duplicate verse
                        verse = None
                    elif are_overlapping_verse_ranges(token.data, verse):
                        verse = merge_verse_ranges(token.data, verse)
                    else:
                        yield from self._create_rows(chapter, verse, verse_text, sentence_start)
                        sentence_start = has_sentence_ending(verse_text)
                        verse_text = ""
                        verse = token.data
                else:
                    verse = token.data
                is_verse_para = True
            elif token.type == UsfmTokenType.PARAGRAPH:
                is_verse_para = _is_verse_para(token)
            elif token.type == UsfmTokenType.NOTE:
                cur_embed_marker = token.marker
                if chapter is not None and verse is not None and self._include_markers:
                    if (
                        prev_token is not None
                        and prev_token.type == UsfmTokenType.PARAGRAPH
                        and _is_verse_para(prev_token)
                    ):
                        verse_text += str(prev_token) + " "
                    verse_text += str(token)
            elif token.type == UsfmTokenType.END:
                assert token.marker is not None
                if cur_embed_marker is not None and token.marker.tag == cur_embed_marker.end_tag:
                    cur_embed_marker = None
                if cur_span_marker is not None and token.marker.tag == cur_span_marker.end_tag:
                    cur_span_marker = None
                if is_verse_para and chapter is not None and verse is not None and self._include_markers:
                    verse_text += str(token)
            elif token.type == UsfmTokenType.CHARACTER:
                assert token.marker is not None
                if token.marker.tag in _SPAN_MARKERS:
                    cur_span_marker = token.marker
                elif token.marker.tag != "qac" and (
                    token.marker.text_type == UsfmTextType.OTHER or token.marker.tag in _EMBED_MARKERS
                ):
                    cur_embed_marker = token.marker
                    if not self._include_markers and token.marker.tag == "rq":
                        verse_text = verse_text.rstrip()
                if is_verse_para and chapter is not None and verse is not None and self._include_markers:
                    if (
                        prev_token is not None
                        and prev_token.type == UsfmTokenType.PARAGRAPH
                        and _is_verse_para(prev_token)
                    ):
                        verse_text += str(prev_token) + " "
                    verse_text += str(token)
                elif _is_table_cell_style(token):
                    if not verse_text[-1].isspace():
                        verse_text += " "
            elif token.type == UsfmTokenType.ATTRIBUTE:
                if is_verse_para and chapter is not None and verse is not None and self._include_markers:
                    verse_text += str(token)
            elif token.type == UsfmTokenType.TEXT:
                token_text = str(token).replace("\r", "").replace("\n", " ")
                if is_verse_para and chapter is not None and verse is not None and token_text != "":
                    if self._include_markers:
                        if (
                            token.text not in {"\r\n", "\n"}
                            and prev_token is not None
                            and prev_token.type == UsfmTokenType.PARAGRAPH
                            and _is_verse_para(prev_token)
                        ):
                            verse_text += str(prev_token)
                        if prev_token is not None and prev_token.type == UsfmTokenType.VERSE:
                            token_text = token_text.lstrip()
                        verse_text += token_text
                    elif cur_embed_marker is None:
                        if (
                            prev_token is not None
                            and prev_token.type == UsfmTokenType.END
                            and (verse_text == "" or verse_text[-1].isspace())
                        ):
                            token_text = token_text.lstrip()
                        verse_text += token_text
            prev_token = token

        if chapter is not None and verse is not None:
            yield from self._create_rows(chapter, verse, verse_text, sentence_start)

    def _read_usfm(self) -> str:
        with self._create_stream_container() as stream_container, TextIOWrapper(
            stream_container.open_stream(), encoding=self._encoding, errors="replace"
        ) as reader:
            return reader.read()


_NONVERSE_PARA_STYLES = {"ms", "mr", "s", "sr", "r", "d", "sp", "rem", "restore", "cl", "cp"}

_SPAN_MARKERS = {"w", "jmp"}

_EMBED_MARKERS = {"fig", "va", "vp", "pro", "rq", "fm"}


def _is_numbered_style(style_prefix: str, style: str) -> bool:
    return style.startswith(style_prefix) and is_integer(style[len(style_prefix) :])


def _is_verse_para(para_token: UsfmToken) -> bool:
    assert para_token.marker is not None
    style = para_token.marker.tag
    if style in _NONVERSE_PARA_STYLES:
        return False

    if _is_numbered_style("ms", style):
        return False

    if _is_numbered_style("s", style):
        return False

    return True


def _is_table_cell_style(char_token: UsfmToken) -> bool:
    assert char_token.marker is not None
    style = char_token.marker.tag
    return (
        _is_numbered_style("th", style)
        or _is_numbered_style("thc", style)
        or _is_numbered_style("thr", style)
        or _is_numbered_style("tc", style)
        or _is_numbered_style("tcc", style)
        or _is_numbered_style("tcr", style)
    )
