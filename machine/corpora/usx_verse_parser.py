import xml.etree.ElementTree as etree
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable, List, Optional

from ..scripture.verse_ref import are_overlapping_verse_ranges
from ..utils.string_utils import has_sentence_ending, is_integer
from .corpora_helpers import merge_verse_ranges
from .usx_token import UsxToken
from .usx_verse import UsxVerse


class UsxVerseParser:
    def parse(self, stream: BinaryIO) -> Iterable[UsxVerse]:
        ctxt = _ParseContext()
        tree = etree.parse(stream)
        root_elem = tree.find(".//book/..")
        if root_elem is None:
            raise RuntimeError("USX does not contain a book element.")
        assert root_elem is not None
        for elem in root_elem:
            if elem.tag == "chapter":
                if ctxt.chapter is not None and ctxt.verse is not None:
                    yield ctxt.create_verse()
                ctxt.chapter = elem.get("number")
                ctxt.verse = None
                ctxt.is_sentence_start = True
            elif elem.tag == "para":
                if not _is_verse_para(elem):
                    ctxt.is_sentence_start = True
                    continue
                ctxt.para_element = elem
                for evt in self._parse_element(elem, ctxt):
                    yield evt

        if ctxt.chapter is not None and ctxt.verse is not None:
            yield ctxt.create_verse()

    def _parse_element(self, elem: etree.Element, ctxt: "_ParseContext") -> Iterable[UsxVerse]:
        if elem.text is not None and ctxt.chapter is not None and ctxt.verse is not None:
            ctxt.add_token(elem.text)
        for e in elem:
            if e.tag == "verse":
                verse = e.get("number")
                if verse is None:
                    verse = e.get("pubnumber")
                assert verse is not None
                if ctxt.chapter is not None and ctxt.verse is not None:
                    if verse == ctxt.verse:
                        yield ctxt.create_verse()

                        # ignore duplicate verse
                        ctxt.verse = None
                    elif are_overlapping_verse_ranges(verse, ctxt.verse):
                        # merge overlapping verse ranges in to one range
                        ctxt.verse = merge_verse_ranges(verse, ctxt.verse)
                    else:
                        yield ctxt.create_verse()
                        ctxt.verse = verse
                else:
                    ctxt.verse = verse
            elif e.tag == "char":
                for evt in self._parse_element(e, ctxt):
                    yield evt
            elif e.tag == "wg":
                if e.text is not None and ctxt.chapter is not None and ctxt.verse is not None:
                    ctxt.add_token(e.text, e)

            if e.tail is not None and ctxt.chapter is not None and ctxt.verse is not None:
                ctxt.add_token(e.tail)


_NONVERSE_PARA_STYLES = {"ms", "mr", "s", "sr", "r", "d", "sp", "rem"}


def _is_numbered_style(style_prefix: str, style: str) -> bool:
    return style.startswith(style_prefix) and is_integer(style[len(style_prefix) :])


def _is_verse_para(para_elem: etree.Element) -> bool:
    style = para_elem.get("style", "")
    if style in _NONVERSE_PARA_STYLES:
        return False

    if _is_numbered_style("ms", style):
        return False

    if _is_numbered_style("s", style):
        return False

    return True


@dataclass
class _ParseContext:
    chapter: Optional[str] = None
    verse: Optional[str] = None
    is_sentence_start: bool = True
    para_element: Optional[etree.Element] = None
    _verse_tokens: List[UsxToken] = field(default_factory=list)

    def add_token(self, text: str, elem: Optional[etree.Element] = None) -> None:
        assert self.para_element is not None
        self._verse_tokens.append(UsxToken(self.para_element, text, elem))

    def create_verse(self) -> UsxVerse:
        assert self.chapter is not None and self.verse is not None
        verse = UsxVerse(self.chapter, self.verse, self.is_sentence_start, self._verse_tokens)
        self.is_sentence_start = has_sentence_ending(verse.text)
        self._verse_tokens.clear()
        return verse
