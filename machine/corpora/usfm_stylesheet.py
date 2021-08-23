from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .usfm_marker import UsfmJustification, UsfmMarker, UsfmStyleType, UsfmTextProperties, UsfmTextType


class UsfmStylesheet:
    def __init__(self, filename: StrPath, alternate_filename: Optional[StrPath] = None) -> None:
        self._markers: Dict[str, UsfmMarker] = {}
        self._parse(Path(filename))
        if alternate_filename is not None:
            self._parse(Path(alternate_filename))

    def get_marker(self, marker_str: str) -> UsfmMarker:
        marker = self._markers.get(marker_str)
        if marker is None:
            marker = self._create_marker(marker_str)
            marker.style_type = UsfmStyleType.UNKNOWN
        return marker

    def _parse(self, filename: Path) -> None:
        if not filename.is_file():
            name = filename.name
            if name == "usfm.sty" or name == "usfm_sb.sty":
                filename = Path(__file__).parent / name
            else:
                raise FileNotFoundError("The stylesheet does not exist.")

        with open(filename, "r", encoding="utf-8-sig") as stream:
            entries = _split_stylesheet(stream)

        for i in range(len(entries)):
            entry_marker, entry_text = entries[i]

            if entry_marker != "marker":
                continue

            parts = entry_text.split()
            if len(parts) > 1 and parts[1] == "-":
                # If the entry looks like "\marker xy -" remove the tag and its end tag if any
                if parts[0] in self._markers:
                    del self._markers[parts[0]]
                if parts[0] + "*" in self._markers:
                    del self._markers[parts[0] + "*"]
                continue

            entry_marker = self._create_marker(entry_text)
            end_marker = _parse_marker_entry(entry_marker, entries, i + 1)

            if end_marker is not None and end_marker.marker not in self._markers:
                self._markers[end_marker.marker] = end_marker

    def _create_marker(self, marker_str: str) -> UsfmMarker:
        # If tag already exists update with addtl info (normally from custom.sty)
        marker = self._markers.get(marker_str)
        if marker is None:
            marker = UsfmMarker(marker_str)
            if marker_str != "c" and marker_str != "v":
                marker.text_properties = UsfmTextProperties.PUBLISHABLE
            self._markers[marker_str] = marker
        return marker


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
        line = line.split("#")[0].strip()
        if line == "":
            continue

        if not line.startswith("\\"):
            # ignore lines that do not start with a backslash
            continue

        parts = line.split(maxsplit=2)
        entries.append((parts[0][1:].lower(), parts[1].strip() if len(parts) > 1 else ""))
    return entries


def _parse_text_properties(marker: UsfmMarker, entry_text: str) -> None:
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


def _parse_text_type(marker: UsfmMarker, entry_text: str) -> None:
    entry_text = entry_text.lower()
    if entry_text == "chapternumber":
        marker.text_properties |= UsfmTextProperties.CHAPTER
    if entry_text == "versenumber":
        marker.text_properties |= UsfmTextProperties.VERSE

    text_type = _TEXT_TYPE_MAPPINGS.get(entry_text)
    if text_type is not None:
        marker.text_type = text_type


def _parse_marker_entry(marker: UsfmMarker, entries: List[Tuple[str, str]], entry_index: int) -> Optional[UsfmMarker]:
    # The following items are present for conformance with Paratext release 5.0 stylesheets.  Release 6.0 and later
    # follows the guidelines set in InitPropertyMaps.

    # Make sure \id gets book property
    if marker.marker == "id":
        marker.text_properties |= UsfmTextProperties.BOOK

    end_marker: Optional[UsfmMarker] = None
    while entry_index < len(entries):
        entry_marker, entry_text = entries[entry_index]
        entry_index += 1

        if entry_marker == "marker":
            break

        if entry_marker == "name":
            marker.name = entry_text
        elif entry_marker == "description":
            marker.description = entry_text
        elif entry_marker == "fontname":
            marker.font_name = entry_text
        elif entry_marker == "fontsize":
            if entry_text == "-":
                marker.font_size = 0
            else:
                font_size = parse_integer(entry_text)
                if font_size is not None and font_size >= 0:
                    marker.font_size = font_size
        elif entry_marker == "xmltag":
            marker.xml_tag = entry_text
        elif entry_marker == "encoding":
            marker.encoding = entry_text
        elif entry_marker == "linespacing":
            line_spacing = parse_integer(entry_text)
            if line_spacing is not None and line_spacing >= 0:
                marker.line_spacing = line_spacing
        elif entry_marker == "spacebefore":
            space_before = parse_integer(entry_text)
            if space_before is not None and space_before >= 0:
                marker.space_before = space_before
        elif entry_marker == "spaceafter":
            space_after = parse_integer(entry_text)
            if space_after is not None and space_after >= 0:
                marker.space_after = space_after
        elif entry_marker == "leftmargin":
            left_margin = parse_integer(entry_text)
            if left_margin is not None and left_margin >= 0:
                marker.left_margin = left_margin
        elif entry_marker == "rightmargin":
            right_margin = parse_integer(entry_text)
            if right_margin is not None and right_margin >= 0:
                marker.right_margin = right_margin
        elif entry_marker == "firstlineindent":
            first_line_indent = parse_integer(entry_text)
            if first_line_indent is not None and first_line_indent >= 0:
                marker.first_line_indent = first_line_indent
        elif entry_marker == "rank":
            if entry_text == "-":
                marker.rank = 0
            else:
                rank = parse_integer(entry_text)
                if rank is not None and rank >= 0:
                    marker.rank = rank
        elif entry_marker == "bold":
            marker.bold = entry_text != "-"
        elif entry_marker == "smallcaps":
            marker.small_caps = entry_text != "-"
        elif entry_marker == "subscript":
            marker.subscript = entry_text != "-"
        elif entry_marker == "italic":
            marker.italic = entry_text != "-"
        elif entry_marker == "regular":
            marker.italic = False
            marker.bold = False
            marker.superscript = False
            marker.regular = True
        elif entry_marker == "underline":
            marker.underline = entry_text != "-"
        elif entry_marker == "superscript":
            marker.superscript = entry_text != "-"
        elif entry_marker == "notrepeatable":
            marker.not_repeatable = entry_text != "-"
        elif entry_marker == "textproperties":
            _parse_text_properties(marker, entry_text)
        elif entry_marker == "texttype":
            _parse_text_type(marker, entry_text)
        elif entry_marker == "color":
            if entry_text == "-":
                marker.color = 0
            else:
                color = parse_integer(entry_text)
                if color is not None and color >= 0:
                    marker.color = color
        elif entry_marker == "justification":
            justification = _JUSTIFICATION_MAPPINGS.get(entry_text.lower())
            if justification is not None:
                marker.justification = justification
        elif entry_marker == "styletype":
            style_type = _STYLE_MAPPINGS.get(entry_text.lower())
            if style_type is not None:
                marker.style_type = style_type
        elif entry_marker == "occursunder":
            marker.occurs_under.update(entry_text.split())
        elif entry_marker == "endmarker":
            end_marker = UsfmMarker(entry_text)
            end_marker.style_type = UsfmStyleType.END
            marker.end_marker = entry_text

    # If we have not seen an end marker but this is a character style
    if marker.style_type == UsfmStyleType.CHARACTER and end_marker is None:
        end_marker_str = marker.marker + "*"
        end_marker = UsfmMarker(end_marker_str)
        end_marker.style_type = UsfmStyleType.END
        marker.end_marker = end_marker_str

    # Special cases
    if (
        marker.text_type == UsfmTextType.OTHER
        and (marker.text_properties & UsfmTextProperties.NONPUBLISHABLE) != UsfmTextProperties.NONPUBLISHABLE
        and (marker.text_properties & UsfmTextProperties.CHAPTER) != UsfmTextProperties.CHAPTER
        and (marker.text_properties & UsfmTextProperties.VERSE) != UsfmTextProperties.VERSE
        and (marker.style_type == UsfmStyleType.CHARACTER or marker.style_type == UsfmStyleType.PARAGRAPH)
    ):
        marker.text_properties |= UsfmTextProperties.PUBLISHABLE

    return end_marker
