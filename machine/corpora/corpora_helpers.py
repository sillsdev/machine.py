import os
import platform
from glob import glob
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, TypeVar

import regex as re

from ..scripture.canon import book_id_to_number
from ..scripture.verse_ref import VERSE_RANGE_SEPARATOR, VERSE_SEQUENCE_INDICATOR, Versification, VersificationType


def get_files(file_patterns: Iterable[str]) -> Iterable[Tuple[str, str]]:
    file_patterns = list(file_patterns)
    if len(file_patterns) == 1 and os.path.isfile(file_patterns[0]):
        yield ("*all*", file_patterns[0])
    else:
        for file_pattern in file_patterns:
            path = file_pattern
            search_pattern = "*"
            if not file_pattern.endswith(os.sep) and not os.path.isdir(file_pattern):
                path = os.path.dirname(file_pattern)
                search_pattern = os.path.basename(file_pattern)

            if path == "":
                path = "."

            base, _ = os.path.splitext(search_pattern)
            converted_mask = re.escape(base).replace("\\*", "(.*)").replace("\\?", "(.)")
            mask_regex = re.compile(converted_mask, re.IGNORECASE if platform.system() == "Windows" else 0)

            for filename in glob(os.path.join(path, search_pattern)):
                id = os.path.basename(filename)
                id, _ = os.path.splitext(id)
                match = mask_regex.fullmatch(id)
                if match is not None:
                    updated_id = ""
                    for group in match.groups():
                        if group is None:
                            continue
                        if len(updated_id) > 0:
                            updated_id += "-"
                        updated_id += group
                    if len(updated_id) > 0:
                        id = updated_id
                yield (id, filename)


T = TypeVar("T")


def gen(iterable: Iterable[T] = []) -> Generator[T, None, None]:
    return (i for i in iterable)


def get_scripture_text_sort_key(id: str) -> str:
    return str(book_id_to_number(id)).zfill(3)


def get_usx_id(filename: Path) -> str:
    name = filename.name
    if len(name) == 3:
        return name
    return name[3:6]


def get_usx_versification(project_dir: Path, versification: Optional[Versification]) -> Versification:
    versification_filename = project_dir / "versification.vrs"
    if versification is None and versification_filename.is_file():
        versification_name = project_dir.name
        versification = Versification.load(versification_filename, fallback_name=versification_name)
    return Versification.get_builtin(VersificationType.ENGLISH) if versification is None else versification


def merge_verse_ranges(verse1: str, verse2: str) -> str:
    text = ""
    verse1_nums = set(_get_verse_nums(verse1))
    verse2_nums = set(_get_verse_nums(verse2))
    start_verse_str = ""
    prev_verse_num = -1
    prev_verse_str = ""
    for verse_num, verse_str in sorted(verse1_nums | verse2_nums, key=lambda x: x[0]):
        if prev_verse_num == -1:
            start_verse_str = verse_str
        elif prev_verse_num != verse_num - 1:
            if len(text) > 0:
                text += VERSE_SEQUENCE_INDICATOR
            text += _get_verse_range(start_verse_str, prev_verse_str)
            start_verse_str = verse_str
        prev_verse_num = verse_num
        prev_verse_str = verse_str
    if len(text) > 0:
        text += VERSE_SEQUENCE_INDICATOR
    text += _get_verse_range(start_verse_str, prev_verse_str)
    return text


def _get_verse_range(start_verse_num: str, end_verse_num: str) -> str:
    verse_range = start_verse_num
    if end_verse_num != start_verse_num:
        verse_range += VERSE_RANGE_SEPARATOR
        verse_range += end_verse_num
    return verse_range


def _get_verse_nums(verse: str) -> Iterable[Tuple[int, str]]:
    parts = verse.split(VERSE_SEQUENCE_INDICATOR)
    for part in parts:
        pieces = part.split(VERSE_RANGE_SEPARATOR)
        start_verse_num = _get_verse_num(pieces[0])
        yield start_verse_num, pieces[0]
        if len(pieces) <= 1:
            continue

        end_verse_num = _get_verse_num(pieces[1])
        for verse_num in range(start_verse_num + 1, end_verse_num):
            yield verse_num, str(verse_num)

        yield end_verse_num, pieces[1]


def _get_verse_num(verse_str: str) -> int:
    v_num = 0
    for ch in verse_str:
        if not ch.isdigit():
            break
        v_num = v_num * 10 + int(ch)
    return v_num
