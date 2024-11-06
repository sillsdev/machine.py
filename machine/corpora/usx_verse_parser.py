from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable, List, Optional
from xml.etree import ElementTree

from ..scripture.verse_ref import are_overlapping_verse_ranges
from ..utils.string_utils import has_sentence_ending
from .corpora_utils import merge_verse_ranges
from .usx_token import UsxToken
from .usx_verse import UsxVerse


class UsxVerseParser:
    def __init__(self, merge_segments: bool = False) -> None:
        self._merge_segments = merge_segments

    def parse(self, stream: BinaryIO) -> Iterable[UsxVerse]:
        ctxt = _ParseContext()
        tree = ElementTree.parse(stream)
        root_elem = tree.find(".//book/..")
        if root_elem is None:
            raise RuntimeError("USX does not contain a book element.")
        assert root_elem is not None
        ctxt.parent_element = root_elem
        for verse in self._parse_element(root_elem, ctxt):
            yield verse

        if ctxt.chapter is not None and ctxt.verse is not None:
            yield ctxt.create_verse()

    def _parse_element(self, elem: ElementTree.Element, ctxt: _ParseContext) -> Iterable[UsxVerse]:
        if elem.text is not None and ctxt.chapter is not None and ctxt.verse is not None:
            ctxt.add_token(elem.text)
        for e in elem:
            if e.tag == "chapter":
                if ctxt.chapter is not None and ctxt.verse is not None:
                    yield ctxt.create_verse()
                ctxt.chapter = e.get("number")
                ctxt.verse = None
                ctxt.is_sentence_start = True
            elif e.tag == "para":
                if not _is_verse_para(e):
                    ctxt.is_sentence_start = True
                    continue
                ctxt.parent_element = e
                for evt in self._parse_element(e, ctxt):
                    yield evt
            elif e.tag == "verse":
                if "eid" in e.attrib:
                    yield ctxt.create_verse()
                    ctxt.verse = None
                else:
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
                if e.get("style") == "rq":
                    if ctxt.chapter is not None and ctxt.verse is not None:
                        ctxt.add_token("", e)
                else:
                    for evt in self._parse_element(e, ctxt):
                        yield evt
            elif e.tag == "wg":
                if e.text is not None and ctxt.chapter is not None and ctxt.verse is not None:
                    ctxt.add_token(e.text, e)
            elif e.tag == "figure":
                if ctxt.chapter is not None and ctxt.verse is not None:
                    ctxt.add_token("", e)
            elif e.tag == "table":
                for evt in self._parse_element(e, ctxt):
                    yield evt
            elif e.tag == "row":
                for evt in self._parse_element(e, ctxt):
                    yield evt
            elif e.tag == "cell":
                ctxt.parent_element = e
                for evt in self._parse_element(e, ctxt):
                    yield evt

            if e.tail is not None and ctxt.chapter is not None and ctxt.verse is not None:
                ctxt.add_token(e.tail)


_VERSE_PARA_STYLES = {
    # Paragraphs
    "p",
    "m",
    "po",
    "pr",
    "cls",
    "pmo",
    "pm",
    "pmc",
    "pmr",
    "pi",
    "pc",
    "mi",
    "nb",
    # Poetry
    "q",
    "qc",
    "qr",
    "qm",
    "qd",
    "b",
    "d",
    # Lists
    "lh",
    "li",
    "lf",
    "lim",
    # Deprecated
    "ph",
    "phi",
    "ps",
    "psi",
}


def _is_verse_para(para_elem: ElementTree.Element) -> bool:
    style = para_elem.get("style", "")
    style = style.rstrip(string.digits)
    return style in _VERSE_PARA_STYLES


@dataclass
class _ParseContext:
    chapter: Optional[str] = None
    verse: Optional[str] = None
    is_sentence_start: bool = True
    parent_element: Optional[ElementTree.Element] = None
    _verse_tokens: List[UsxToken] = field(default_factory=list)

    def add_token(self, text: str, elem: Optional[ElementTree.Element] = None) -> None:
        assert self.parent_element is not None
        self._verse_tokens.append(UsxToken(self.parent_element, text, elem))

    def create_verse(self) -> UsxVerse:
        assert self.chapter is not None and self.verse is not None
        verse = UsxVerse(self.chapter, self.verse, self.is_sentence_start, self._verse_tokens)
        self.is_sentence_start = has_sentence_ending(verse.text)
        self._verse_tokens.clear()
        return verse
