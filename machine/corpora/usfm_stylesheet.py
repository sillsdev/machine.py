from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import regex as re

from ..utils.file_utils import detect_encoding
from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .usfm_tag import UsfmJustification, UsfmStyleAttribute, UsfmStyleType, UsfmTag, UsfmTextProperties, UsfmTextType

_CELL_RANGE_REGEX = re.compile(r"^(t[ch][cr]?[1-5])-([2-5])$")


def is_cell_range(marker: str) -> Tuple[bool, str, int]:
    match = _CELL_RANGE_REGEX.match(marker)
    if match is not None:
        base_tag = match.group(1)
        col_span = int(match.group(2)[0]) - int(base_tag[-1]) + 1
        if col_span >= 2:
            return True, base_tag, col_span

    return False, "", 0


class UsfmStylesheet:
    def __init__(self, filename: StrPath, alternate_filename: Optional[StrPath] = None) -> None:
        self._tags: Dict[str, UsfmTag] = {}
        self._parse(filename)
        if alternate_filename is not None:
            try:
                self._parse(alternate_filename)
            except UnicodeDecodeError:
                encoding = detect_encoding(alternate_filename)
                self._parse(alternate_filename, encoding)

    def get_tag(self, marker: str) -> UsfmTag:
        tag = self._tags.get(marker)
        if tag is not None:
            return tag

        is_cell, base_marker, _ = is_cell_range(marker)
        if is_cell:
            tag = self._tags.get(base_marker)
            if tag is not None:
                return tag

        tag = self._create_tag(marker)
        tag.style_type = UsfmStyleType.UNKNOWN
        return tag

    def _parse(self, filename: StrPath, encoding: str = "utf-8-sig") -> None:
        if not isinstance(filename, Path):
            filename = Path(filename)
        if not filename.is_file():
            name = filename.name
            if name == "usfm.sty" or name == "usfm_sb.sty":
                filename = Path(__file__).parent / name
            else:
                raise FileNotFoundError("The stylesheet does not exist.")

        with filename.open("r", encoding=encoding) as stream:
            entries = _split_stylesheet(stream)

        for i in range(len(entries)):
            entry_marker, entry_text = entries[i]

            if entry_marker != "marker":
                continue

            parts = entry_text.split()
            if len(parts) > 1 and parts[1] == "-":
                # If the entry looks like "\marker xy -" remove the tag and its end tag if any
                if parts[0] in self._tags:
                    del self._tags[parts[0]]
                if parts[0] + "*" in self._tags:
                    del self._tags[parts[0] + "*"]
                continue

            entry_marker = self._create_tag(entry_text)
            end_marker = _parse_tag_entry(entry_marker, entries, i + 1)

            if end_marker is not None and end_marker.marker not in self._tags:
                self._tags[end_marker.marker] = end_marker

    def _create_tag(self, marker: str) -> UsfmTag:
        # If tag already exists update with addtl info (normally from custom.sty)
        tag = self._tags.get(marker)
        if tag is None:
            tag = UsfmTag(marker)
            if marker != "c" and marker != "v":
                tag.text_properties = UsfmTextProperties.PUBLISHABLE
            self._tags[marker] = tag
        return tag


_JUSTIFICATION_MAPPINGS = {
    "left": UsfmJustification.LEFT,
    "center": UsfmJustification.CENTER,
    "right": UsfmJustification.RIGHT,
    "both": UsfmJustification.BOTH,
}

_STYLE_MAPPINGS = {
    "character": UsfmStyleType.CHARACTER,
    "paragraph": UsfmStyleType.PARAGRAPH,
    "note": UsfmStyleType.NOTE,
}

_TEXT_TYPE_MAPPINGS = {
    "title": UsfmTextType.TITLE,
    "section": UsfmTextType.SECTION,
    "versetext": UsfmTextType.VERSE_TEXT,
    "notetext": UsfmTextType.NOTE_TEXT,
    "other": UsfmTextType.OTHER,
    "backtranslation": UsfmTextType.BACK_TRANSLATION,
    "translationnote": UsfmTextType.TRANSLATION_NOTE,
    "versenumber": UsfmTextType.VERSE_TEXT,
    "chapternumber": UsfmTextType.OTHER,
}

_TEXT_PROPERTY_MAPPINGS = {
    "verse": UsfmTextProperties.VERSE,
    "chapter": UsfmTextProperties.CHAPTER,
    "paragraph": UsfmTextProperties.PARAGRAPH,
    "publishable": UsfmTextProperties.PUBLISHABLE,
    "vernacular": UsfmTextProperties.VERNACULAR,
    "poetic": UsfmTextProperties.POETIC,
    "level_1": UsfmTextProperties.LEVEL1,
    "level_2": UsfmTextProperties.LEVEL2,
    "level_3": UsfmTextProperties.LEVEL3,
    "level_4": UsfmTextProperties.LEVEL4,
    "level_5": UsfmTextProperties.LEVEL5,
    "crossreference": UsfmTextProperties.CROSS_REFERENCE,
    "nonpublishable": UsfmTextProperties.NONPUBLISHABLE,
    "nonvernacular": UsfmTextProperties.NONVERNACULAR,
    "book": UsfmTextProperties.BOOK,
    "note": UsfmTextProperties.NOTE,
}


def _split_stylesheet(stream: TextIO) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for line in stream:
        if line.startswith("#!"):
            line = line[2:]
        line = line.split("#")[0].strip()
        if line == "":
            continue

        if not line.startswith("\\"):
            # ignore lines that do not start with a backslash
            continue

        parts = line.split(maxsplit=2)
        entries.append((parts[0][1:].lower(), parts[1].strip() if len(parts) > 1 else ""))
    return entries


def _parse_text_properties(marker: UsfmTag, entry_text: str) -> None:
    entry_text = entry_text.lower()
    parts = entry_text.split()

    for part in parts:
        if part.strip() == "":
            continue

        text_property = _TEXT_PROPERTY_MAPPINGS.get(part)
        if text_property is not None:
            marker.text_properties |= text_property

    if (marker.text_properties & UsfmTextProperties.NONPUBLISHABLE) == UsfmTextProperties.NONPUBLISHABLE:
        marker.text_properties &= ~UsfmTextProperties.PUBLISHABLE


def _parse_text_type(marker: UsfmTag, entry_text: str) -> None:
    entry_text = entry_text.lower()
    if entry_text == "chapternumber":
        marker.text_properties |= UsfmTextProperties.CHAPTER
    if entry_text == "versenumber":
        marker.text_properties |= UsfmTextProperties.VERSE

    text_type = _TEXT_TYPE_MAPPINGS.get(entry_text)
    if text_type is not None:
        marker.text_type = text_type


def _parse_attributes(marker: UsfmTag, entry_text: str) -> None:
    attribute_names = entry_text.split()
    if len(attribute_names) == 0:
        raise ValueError("Attributes cannot be empty.")
    found_optional = False
    for attribute in attribute_names:
        is_optional = attribute.startswith("?")
        if not is_optional and found_optional:
            raise ValueError("Required attributes must precede optional attributes.")

        marker.attributes.append(UsfmStyleAttribute(attribute[1:] if is_optional else attribute, not is_optional))
        found_optional |= is_optional

    if sum(1 for a in marker.attributes if a.is_required) <= 1:
        marker.default_attribute_name = marker.attributes[0].name


def _parse_tag_entry(tag: UsfmTag, entries: List[Tuple[str, str]], entry_index: int) -> Optional[UsfmTag]:
    # The following items are present for conformance with Paratext release 5.0 stylesheets.  Release 6.0 and later
    # follows the guidelines set in InitPropertyMaps.

    # Make sure \id gets book property
    if tag.marker == "id":
        tag.text_properties |= UsfmTextProperties.BOOK

    end_marker: Optional[UsfmTag] = None
    while entry_index < len(entries):
        entry_marker, entry_text = entries[entry_index]
        entry_index += 1

        if entry_marker == "marker":
            break

        if entry_marker == "name":
            tag.name = entry_text
        elif entry_marker == "description":
            tag.description = entry_text
        elif entry_marker == "fontname":
            tag.font_name = entry_text
        elif entry_marker == "fontsize":
            if entry_text == "-":
                tag.font_size = 0
            else:
                font_size = parse_integer(entry_text)
                if font_size is not None and font_size >= 0:
                    tag.font_size = font_size
        elif entry_marker == "xmltag":
            tag.xml_tag = entry_text
        elif entry_marker == "encoding":
            tag.encoding = entry_text
        elif entry_marker == "linespacing":
            line_spacing = parse_integer(entry_text)
            if line_spacing is not None and line_spacing >= 0:
                tag.line_spacing = line_spacing
        elif entry_marker == "spacebefore":
            space_before = parse_integer(entry_text)
            if space_before is not None and space_before >= 0:
                tag.space_before = space_before
        elif entry_marker == "spaceafter":
            space_after = parse_integer(entry_text)
            if space_after is not None and space_after >= 0:
                tag.space_after = space_after
        elif entry_marker == "leftmargin":
            left_margin = parse_integer(entry_text)
            if left_margin is not None and left_margin >= 0:
                tag.left_margin = left_margin
        elif entry_marker == "rightmargin":
            right_margin = parse_integer(entry_text)
            if right_margin is not None and right_margin >= 0:
                tag.right_margin = right_margin
        elif entry_marker == "firstlineindent":
            first_line_indent = parse_integer(entry_text)
            if first_line_indent is not None and first_line_indent >= 0:
                tag.first_line_indent = first_line_indent
        elif entry_marker == "rank":
            if entry_text == "-":
                tag.rank = 0
            else:
                rank = parse_integer(entry_text)
                if rank is not None and rank >= 0:
                    tag.rank = rank
        elif entry_marker == "bold":
            tag.bold = entry_text != "-"
        elif entry_marker == "smallcaps":
            tag.small_caps = entry_text != "-"
        elif entry_marker == "subscript":
            tag.subscript = entry_text != "-"
        elif entry_marker == "italic":
            tag.italic = entry_text != "-"
        elif entry_marker == "regular":
            tag.italic = False
            tag.bold = False
            tag.superscript = False
            tag.regular = True
        elif entry_marker == "underline":
            tag.underline = entry_text != "-"
        elif entry_marker == "superscript":
            tag.superscript = entry_text != "-"
        elif entry_marker == "notrepeatable":
            tag.not_repeatable = entry_text != "-"
        elif entry_marker == "textproperties":
            _parse_text_properties(tag, entry_text)
        elif entry_marker == "texttype":
            _parse_text_type(tag, entry_text)
        elif entry_marker == "color":
            if entry_text == "-":
                tag.color = 0
            else:
                color = parse_integer(entry_text)
                if color is not None and color >= 0:
                    tag.color = color
        elif entry_marker == "justification":
            justification = _JUSTIFICATION_MAPPINGS.get(entry_text.lower())
            if justification is not None:
                tag.justification = justification
        elif entry_marker == "styletype":
            style_type = _STYLE_MAPPINGS.get(entry_text.lower())
            if style_type is not None:
                tag.style_type = style_type
        elif entry_marker == "occursunder":
            tag.occurs_under.update(entry_text.split())
        elif entry_marker == "endmarker":
            end_marker = UsfmTag(entry_text)
            end_marker.style_type = UsfmStyleType.END
            tag.end_marker = entry_text
        elif entry_marker == "attributes":
            _parse_attributes(tag, entry_text)

    # If we have not seen an end marker but this is a character style
    if tag.style_type == UsfmStyleType.CHARACTER and end_marker is None:
        end_marker_str = tag.marker + "*"
        end_marker = UsfmTag(end_marker_str)
        end_marker.style_type = UsfmStyleType.END
        tag.end_marker = end_marker_str

    # Special cases
    if (
        tag.text_type == UsfmTextType.OTHER
        and (tag.text_properties & UsfmTextProperties.NONPUBLISHABLE) != UsfmTextProperties.NONPUBLISHABLE
        and (tag.text_properties & UsfmTextProperties.CHAPTER) != UsfmTextProperties.CHAPTER
        and (tag.text_properties & UsfmTextProperties.VERSE) != UsfmTextProperties.VERSE
        and (tag.style_type == UsfmStyleType.CHARACTER or tag.style_type == UsfmStyleType.PARAGRAPH)
    ):
        tag.text_properties |= UsfmTextProperties.PUBLISHABLE

    return end_marker
