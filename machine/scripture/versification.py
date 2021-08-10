from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO, Tuple, Union

import regex

from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .canon import book_id_to_number, is_canonical
from .verse_ref import VerseRef, get_bbbcccvvv


class VersificationType(IntEnum):
    UNKNOWN = 0
    ORIGINAL = 1
    SEPTUAGINT = 2
    VULGATE = 3
    ENGLISH = 4
    RUSSIAN_PROTESTANT = 5
    RUSSIAN_ORTHODOX = 6


class Versification:
    _BUILTIN_VERSIFICATIONS: Dict[VersificationType, "Versification"] = {}
    _BUILTIN_VERSIFICATION_FILENAMES = {
        VersificationType.ORIGINAL: "org.vrs.txt",
        VersificationType.ENGLISH: "eng.vrs.txt",
        VersificationType.SEPTUAGINT: "lxx.vrs.txt",
        VersificationType.VULGATE: "vul.vrs.txt",
        VersificationType.RUSSIAN_ORTHODOX: "rso.vrs.txt",
        VersificationType.RUSSIAN_PROTESTANT: "rsc.vrs.txt",
    }
    _BUILTIN_VERSIFICATION_NAMES_TO_TYPES = {
        "Original": VersificationType.ORIGINAL,
        "Septuagint": VersificationType.SEPTUAGINT,
        "Vulgate": VersificationType.VULGATE,
        "English": VersificationType.ENGLISH,
        "RussianProtestant": VersificationType.RUSSIAN_PROTESTANT,
        "RussianOrthodox": VersificationType.RUSSIAN_ORTHODOX,
    }
    _NON_CANONICAL_LAST_CHAPTER_OR_VERSE = 998

    @classmethod
    def create(cls, name: str) -> "Versification":
        type = cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES.get(name)
        if type is not None:
            return cls.get_builtin(type)

        versification = Versification(name)
        with open(Path(__file__).parent / "eng.vrs.txt", "r", encoding="utf-8") as file:
            versification = cls.parse(file, versification=versification)
        return versification

    @classmethod
    def get_builtin(cls, type: Union[VersificationType, int, str]) -> "Versification":
        if isinstance(type, int):
            type = VersificationType(type)
        elif isinstance(type, str):
            if type in cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES:
                type = cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES[type]
            else:
                type = VersificationType[type]

        versification = cls._BUILTIN_VERSIFICATIONS.get(type)
        if versification is not None:
            return versification

        filename = cls._BUILTIN_VERSIFICATION_FILENAMES.get(type)
        if filename is None:
            raise ValueError("The versification type is unknown.")

        path = Path(__file__).parent / filename

        with open(path, "r", encoding="utf-8") as file:
            versification = cls.parse(file)
        cls._BUILTIN_VERSIFICATIONS[type] = versification
        return versification

    @classmethod
    def load(
        cls,
        filename: StrPath,
        base_versification: Optional["Versification"] = None,
        fallback_name: Optional[str] = None,
    ) -> "Versification":
        with open(filename, "r", encoding="utf-8") as file:
            versification = (
                None
                if base_versification is None or fallback_name is None
                else Versification(fallback_name, filename, base_versification)
            )
            return cls.parse(file, filename, versification, fallback_name)

    @classmethod
    def parse(
        cls,
        stream: TextIO,
        filename: Optional[StrPath] = None,
        versification: Optional["Versification"] = None,
        fallback_name: Optional[str] = None,
    ) -> "Versification":
        return _parse_versification(stream, filename, versification, fallback_name)

    def __init__(
        self, name: str, filename: Optional[StrPath] = None, base_versification: Optional["Versification"] = None
    ) -> None:
        self._name = name
        self._type = VersificationType.UNKNOWN
        if base_versification is None:
            type = Versification._BUILTIN_VERSIFICATION_NAMES_TO_TYPES.get(name)
            if type is not None:
                self._type = type
        self._filename = None if filename is None else Path(filename)
        self._base_versification = base_versification

        self._mappings = _VerseMappings()
        self._excluded_verses: Set[int] = set()
        self._book_list: List[List[int]] = []
        self._verse_segments: Dict[int, Set[str]] = {}
        self.description: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def filename(self) -> Optional[Path]:
        return self._filename

    @property
    def type(self) -> "VersificationType":
        return self._type

    @property
    def base_versification(self) -> Optional["Versification"]:
        return self._base_versification

    def get_last_book(self) -> int:
        return len(self._book_list)

    def get_last_chapter(self, book_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return Versification._NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self._book_list):
            return 1

        chapters = self._book_list[book_num - 1]
        return len(chapters)

    def get_last_verse(self, book_num: int, chapter_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return Versification._NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self._book_list):
            return 1

        chapters = self._book_list[book_num - 1]
        if chapter_num > len(chapters) or chapter_num < 1:
            return 1

        return chapters[chapter_num - 1]

    def first_included_verse(self, book_num: int, chapter_num: int) -> Optional[VerseRef]:
        while True:
            last_verse = self.get_last_verse(book_num, chapter_num)
            for verse_num in range(1, last_verse + 1):
                if not self.is_excluded(get_bbbcccvvv(book_num, chapter_num, verse_num)):
                    return VerseRef(book_num, chapter_num, verse_num, self)

            # Non-excluded verse not found in this chapter, so try next chapter
            chapter_num += 1
            if chapter_num > self.get_last_chapter(book_num):
                break

        return None

    def is_excluded(self, bbbcccvvv: int) -> bool:
        return bbbcccvvv in self._excluded_verses

    def verse_segments(self, bbbcccvvv: int) -> Optional[Set[str]]:
        return self._verse_segments.get(bbbcccvvv)

    def change_versification(self, vref: VerseRef) -> bool:
        if vref.has_multiple:
            return self._change_versification_with_ranges(vref)

        if vref.versification == NULL_VERSIFICATION:
            vref.versification = self
            return True

        orig_versification = vref.versification

        # Map from existing to standard versification
        orig_vref = vref.copy()
        orig_vref.versification = NULL_VERSIFICATION

        standard_vref = orig_versification._mappings.get_standard(orig_vref)
        if standard_vref is None:
            standard_vref = orig_vref

        # If both versifications contain this verse and map this verse to the same location then no versification
        # change is needed.
        standard_vref_this_versification = self._mappings.get_standard(orig_vref)
        if standard_vref_this_versification is None:
            standard_vref_this_versification = orig_vref

        # ESG is a special case since we have added mappings from verses to LXX segments in several versifications and
        # want this mapping to work both ways.
        if (
            vref.book != "ESG"
            and standard_vref == standard_vref_this_versification
            and self._book_chapter_verse_exists(vref)
        ):
            vref.versification = self
            return True

        # Map from standard versification to this versification
        new_vref = self._mappings.get_versification(standard_vref)
        if new_vref is None:
            new_vref = standard_vref

        # If verse has changed, parse new value
        if orig_vref != new_vref:
            vref.copy_from(new_vref)

        vref.versification = self
        return True

    def __eq__(self, other: "Versification") -> bool:
        if self is other:
            return True
        return (
            self._name == other._name
            and self.description == other.description
            and self._book_list == other._book_list
            and self._excluded_verses == other._excluded_verses
            and self._verse_segments == other._verse_segments
            and self._mappings == other._mappings
        )

    def _book_chapter_verse_exists(self, vref: VerseRef) -> bool:
        return (
            vref.book_num <= self.get_last_book()
            and vref.chapter_num <= self.get_last_chapter(vref.book_num)
            and vref.verse_num <= self.get_last_verse(vref.book_num, vref.chapter_num)
        )

    def _change_versification_with_ranges(self, vref: VerseRef) -> bool:
        parts: List[str] = regex.split(r"([,\-])", vref.verse)

        new_vref = vref.copy()
        new_vref.verse = parts[0]
        self.change_versification(new_vref)
        all_same_chapter = True

        for i in range(2, len(parts), 2):
            part_vref = vref.copy()
            part_vref.verse = parts[i]
            self.change_versification(part_vref)
            if new_vref.chapter_num != part_vref.chapter_num:
                all_same_chapter = False
            new_vref.verse = new_vref.verse + parts[i - 1] + part_vref.verse

        vref.copy_from(new_vref)

        return all_same_chapter


class _VerseMappings:
    def __init__(self) -> None:
        self._versification_to_standard: Dict[VerseRef, VerseRef] = {}
        self._standard_to_versification: Dict[VerseRef, VerseRef] = {}

    def add_mapping(self, versification_ref: VerseRef, standard_ref: VerseRef) -> None:
        if sum(1 for _ in versification_ref.all_verses()) != 1 or sum(1 for _ in standard_ref.all_verses()) != 1:
            raise ValueError("Mappings must resolve into a single reference on boths sides.")

        self._versification_to_standard[versification_ref] = standard_ref
        self._standard_to_versification[standard_ref] = versification_ref

    def add_mappings(self, versification_refs: List[VerseRef], standard_refs: List[VerseRef]) -> None:
        for versification_ref in versification_refs:
            for standard_ref in standard_refs:
                self.add_mapping(versification_ref, standard_ref)

    def get_versification(self, standard_ref: VerseRef) -> Optional[VerseRef]:
        return self._standard_to_versification.get(standard_ref)

    def get_standard(self, versification_ref: VerseRef) -> Optional[VerseRef]:
        return self._versification_to_standard.get(versification_ref)

    def __eq__(self, other: "_VerseMappings") -> bool:
        return (
            self._versification_to_standard == other._versification_to_standard
            and self._standard_to_versification == other._standard_to_versification
        )


NULL_VERSIFICATION = Versification("NULL")

_VERSIFICATION_NAME_REGEX = regex.compile(r"#\s*Versification\s+\"(?<name>[^\"]+)\"\s*")


class _LineType(Enum):
    COMMENT = auto()
    CHAPTER_VERSE = auto()
    STANDARD_MAPPING = auto()
    ONE_TO_MANY_MAPPING = auto()
    EXCLUDED_VERSE = auto()
    VERSE_SEGMENTS = auto()


@dataclass(frozen=True)
class _VersificationLine:
    type: _LineType
    line: str
    comment: str
    line_num: int

    def __repr__(self) -> str:
        if self.type == _LineType.CHAPTER_VERSE:
            return self.line
        elif self.type == _LineType.ONE_TO_MANY_MAPPING:
            return f"#! {self.line}"
        elif self.type == _LineType.COMMENT:
            if self.comment != "":
                return f"# {self.comment}"
            return ""
        else:
            if self.comment == "":
                return self.line
            return f"{self.line} # {self.comment}"


def _syntax_error(message: str, line_num: int) -> RuntimeError:
    return RuntimeError(f"Invalid versification syntax: {message}, line: {line_num}.")


def _parse_versification(
    stream: TextIO,
    filename: Optional[StrPath] = None,
    versification: Optional["Versification"] = None,
    fallback_name: Optional[str] = None,
) -> Versification:
    line_num = 1
    for line in stream:
        line = line.strip()
        if versification is None:
            name = ""
            match = _VERSIFICATION_NAME_REGEX.match(line)
            if match is not None:
                name = match.group("name")
            if name != "":
                versification = Versification(name, filename)

        parsed_line = _parse_line(line, line_num)
        if parsed_line.type == _LineType.COMMENT:
            line_num += 1
            continue

        if versification is None:
            if fallback_name is None or fallback_name == "":
                raise _syntax_error("the versification is missing a name", parsed_line.line_num)
            versification = Versification(fallback_name, filename)

        if parsed_line.type == _LineType.CHAPTER_VERSE:
            _parse_chapter_verse_line(versification, parsed_line)
        elif parsed_line.type == _LineType.STANDARD_MAPPING:
            _parse_mapping_line(versification, parsed_line)
        elif parsed_line.type == _LineType.ONE_TO_MANY_MAPPING:
            _parse_range_to_one_mapping_line(versification, parsed_line)
        elif parsed_line.type == _LineType.EXCLUDED_VERSE:
            _parse_excluded_verse(versification, parsed_line)
        elif parsed_line.type == _LineType.VERSE_SEGMENTS:
            if parsed_line.line.find("#") != -1:
                raise _syntax_error("invalid verse segments line", parsed_line.line_num)
            _parse_verse_segments_line(versification, parsed_line, is_builtin=filename is None)
        line_num += 1

    assert versification is not None
    return versification


def _parse_line(line: str, line_num: int) -> _VersificationLine:
    is_comment_line = len(line) > 0 and line[0] == "#"
    parts = line.split("#", maxsplit=2)
    line = parts[0].strip()
    comment = parts[1].strip() if len(parts) == 2 else ""

    if line == "" and len(comment) > 2 and comment[0] == "!":
        # found Paratext 7.3(+) versification line beginning with #!
        line = comment[1:].strip()
        comment = ""
        is_comment_line = False

    if len(line) == 0 or is_comment_line:
        line_type = _LineType.COMMENT
    elif "=" in line:
        # mapping one verse to multiple
        line_type = _LineType.ONE_TO_MANY_MAPPING if line[0] == "&" else _LineType.STANDARD_MAPPING
    elif line[0] == "-":
        line_type = _LineType.EXCLUDED_VERSE
    elif line[0] == "*":
        line_type = _LineType.VERSE_SEGMENTS
    else:
        line_type = _LineType.CHAPTER_VERSE

    return _VersificationLine(line_type, line, comment, line_num)


def _parse_chapter_verse_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    parts = parsed_line.line.split()
    book_num = book_id_to_number(parts[0])
    if book_num == 0:
        raise _syntax_error("invalid book", parsed_line.line_num)

    while len(versification._book_list) < book_num:
        versification._book_list.append([1])

    verses_in_chapter = versification._book_list[book_num - 1].copy()

    chapter = 0
    for part in parts[1:]:
        # END is used if the number of chapters in custom is less than base
        if part == "END":
            if len(verses_in_chapter) > chapter:
                del verses_in_chapter[chapter:]
            break

        pieces = part.split(":")
        c = parse_integer(pieces[0])
        if c is None or c <= 0:
            raise _syntax_error("invalid chapter", parsed_line.line_num)
        chapter = c

        if len(pieces) != 2:
            raise _syntax_error("missing verse", parsed_line.line_num)

        verse_count = parse_integer(pieces[1])
        if verse_count is None or verse_count < 0:
            raise _syntax_error("invalid verse", parsed_line.line_num)

        if len(verses_in_chapter) < chapter:
            for i in range(len(verses_in_chapter), chapter):
                # by default, chapters have one verse
                verses_in_chapter.append(1)
        verses_in_chapter[chapter - 1] = verse_count

    versification._book_list[book_num - 1] = verses_in_chapter


def _parse_mapping_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    try:
        parts = parsed_line.line.split("=")
        left_pieces = parts[0].strip().split("-")
        right_pieces = parts[1].strip().split("-")

        new_verse_ref = VerseRef.from_string(left_pieces[0], NULL_VERSIFICATION)
        left_limit = 0 if len(left_pieces) == 1 else int(left_pieces[1])

        standard_verse_ref = VerseRef.from_string(right_pieces[0], NULL_VERSIFICATION)

        while True:
            versification._mappings.add_mapping(new_verse_ref.copy(), standard_verse_ref.copy())

            if new_verse_ref.verse_num >= left_limit:
                break

            new_verse_ref.verse_num += 1
            standard_verse_ref.verse_num += 1
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)


def _get_references(verse_pieces: List[str]) -> List[VerseRef]:
    if len(verse_pieces) == 1:
        return [VerseRef.from_string(verse_pieces[0], NULL_VERSIFICATION)]

    new_verse_ref = VerseRef.from_string(verse_pieces[0], NULL_VERSIFICATION)
    limit = int(verse_pieces[1])

    verse_refs: List[VerseRef] = []
    while True:
        verse_refs.append(new_verse_ref.copy())
        if new_verse_ref.verse_num >= limit:
            break

        new_verse_ref.verse_num += 1

    return verse_refs


def _parse_range_to_one_mapping_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    line = parsed_line.line[1:]
    try:
        parts = line.split("=")
        left_pieces = parts[0].strip().split("-")
        right_pieces = parts[1].strip().split("-")

        versification_refs = _get_references(left_pieces)
        standard_refs = _get_references(right_pieces)
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)

    # either versification or standard must have just one verse
    if len(versification_refs) != 1 and len(standard_refs) != 1:
        raise _syntax_error("invalid many-to-one mapping", parsed_line.line_num)

    versification._mappings.add_mappings(versification_refs, standard_refs)


def _get_verse_reference(parts: List[str], line_num: int) -> Tuple[str, int, int]:
    book_name = parts[0][1:]
    if book_id_to_number(book_name) == 0:
        raise _syntax_error("invalid book", line_num)

    pieces = parts[1].split(":")
    chapter = int(pieces[0])
    verse = int(pieces[1])
    return book_name, chapter, verse


def _parse_excluded_verse(versification: Versification, parsed_line: _VersificationLine) -> None:
    if (
        len(parsed_line.line) < 8
        or parsed_line.line[0] != "-"
        or ":" not in parsed_line.line
        or " " not in parsed_line.line
    ):
        raise _syntax_error("invalid excluded verse line", parsed_line.line_num)

    parts = parsed_line.line.split()
    try:
        book_name, chapter, verse = _get_verse_reference(parts, parsed_line.line_num)

        verse_ref = VerseRef(book_name, str(chapter), str(verse), versification)
        if verse_ref.bbbcccvvv not in versification._excluded_verses:
            versification._excluded_verses.add(verse_ref.bbbcccvvv)
        else:
            raise _syntax_error("duplicate excluded verse", parsed_line.line_num)
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)


def _remove_spaces(line: str, index: int) -> str:
    if index < 1:
        raise ValueError("Invalid length.")
    if len(line) < 2:
        raise ValueError("Invalid line.")

    result = line[:index]
    parts = line[index:].split()

    for part in parts:
        result += part
    return result


def _parse_verse_segments_line(versification: Versification, parsed_line: _VersificationLine, is_builtin: bool) -> None:
    if (
        len(parsed_line.line) < 8
        or parsed_line.line[0] != "*"
        or ":" not in parsed_line.line
        or " " not in parsed_line.line
        or "," not in parsed_line.line
    ):
        raise _syntax_error("invalid verse segments line", parsed_line.line_num)

    index_of_colon = parsed_line.line.find(":")
    line = _remove_spaces(parsed_line.line, index_of_colon)

    parts = line.split()
    try:
        # Get segmenting information
        segment_start = parts[1].find(",")
        if segment_start == -1:
            raise _syntax_error("invalid segment", parsed_line.line_num)

        segments = parts[1][segment_start + 1 :]

        # Get Scripture reference, throwing an exception if it is not valid.
        parts[1] = parts[1][:segment_start]
        # Remove segment info from chapter:verse reference
        book_name, chapter, verse = _get_verse_reference(parts, parsed_line.line_num)

        segment_set: Set[str] = set()
        nonempty_segment_found = False
        for seg in segments.split(","):
            if seg == "":
                continue
            if nonempty_segment_found and seg == "-":
                raise _syntax_error("unspecified segment location", parsed_line.line_num)

            if seg == "-":
                # '-' indicates no marking for segment
                segment_set.add("")
            else:
                segment_set.add(seg)
                nonempty_segment_found = True

        if len(segment_set) == 1 and next(iter(segment_set)) == "":
            raise _syntax_error("no segments defined", parsed_line.line_num)

        bbbcccvvv = get_bbbcccvvv(book_id_to_number(book_name), chapter, verse)
        # Don't allow overwrites for built-in versifications
        if is_builtin and bbbcccvvv in versification._verse_segments:
            raise _syntax_error("duplicate segment", parsed_line.line_num)

        versification._verse_segments[bbbcccvvv] = segment_set
    except ValueError as e:
        raise _syntax_error("invalid verse reference " + str(e), parsed_line.line_num)
