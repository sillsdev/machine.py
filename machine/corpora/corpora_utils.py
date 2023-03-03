import os
import platform
from glob import glob
from pathlib import Path
from random import Random
from typing import Any, Callable, Generator, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar
from zipfile import ZipFile, ZipInfo

import regex as re

from ..scripture.canon import book_id_to_number
from ..scripture.verse_ref import VERSE_RANGE_SEPARATOR, VERSE_SEQUENCE_INDICATOR, Versification, VersificationType

T = TypeVar("T")


def batch(iterable: Iterable[T], batch_size: int) -> Iterable[Sequence[T]]:
    if isinstance(iterable, Sequence) and len(iterable) <= batch_size:
        yield iterable

    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def get_split_indices(
    corpus_size: int, percent: Optional[float] = None, size: Optional[int] = None, seed: Any = None
) -> Set[int]:
    if percent is None and size is None:
        percent = 0.1

    if percent is not None:
        split_size = int(percent * corpus_size)
        if size is not None:
            split_size = min(split_size, size)
    else:
        assert size is not None
        split_size = size

    rand = Random()
    if seed is not None:
        rand.seed(seed)
    return set(rand.sample(range(corpus_size), min(split_size, corpus_size)))


def get_files(file_patterns: Iterable[str]) -> Iterable[Tuple[str, str]]:
    file_patterns = list(file_patterns)
    if len(file_patterns) == 1 and os.path.isfile(file_patterns[0]):
        yield ("*all*", file_patterns[0])
    else:
        for file_pattern in file_patterns:
            if "*" not in file_pattern and "?" not in file_pattern and not os.path.exists(file_pattern):
                raise FileNotFoundError(f"The specified path does not exist: {file_pattern}.")

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


def gen(iterable: Iterable[T] = []) -> Generator[T, None, None]:
    return (i for i in iterable)


def get_scripture_text_sort_key(id: str) -> str:
    return str(book_id_to_number(id)).zfill(3)


def get_usx_id(filename: Path) -> str:
    name = filename.stem
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
    for verse_num, verse_str in sorted(verse1_nums | verse2_nums):
        if prev_verse_num == -1:
            start_verse_str = verse_str
        elif prev_verse_num == verse_num and str(prev_verse_num) == prev_verse_str and str(verse_num) != verse_str:
            # the verse segment is subsumed by the previous full verse, so skip it
            continue
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


_ENCODINGS = {
    37: "cp037",
    437: "cp437",
    500: "cp500",
    720: "cp720",
    737: "cp737",
    775: "cp775",
    850: "cp850",
    852: "cp852",
    855: "cp855",
    857: "cp857",
    858: "cp858",
    860: "cp860",
    861: "cp861",
    862: "cp862",
    863: "cp863",
    864: "cp864",
    865: "cp865",
    866: "cp866",
    869: "cp869",
    874: "cp874",
    875: "cp875",
    932: "cp932",
    936: "gb2313",
    949: "cp949",
    950: "cp950",
    1026: "cp1026",
    1140: "cp1140",
    1200: "utf_16",
    1201: "utf_16_be",
    1250: "cp1250",
    1251: "cp1251",
    1252: "cp1252",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    1361: "johab",
    12000: "utf_32",
    12001: "utf_32_be",
    20127: "ascii",
    20936: "gb2312",
    28591: "latin_1",
    28598: "iso8859_8",
    50220: "iso2022_jp",
    50225: "iso2022_kr",
    51932: "euc_jp",
    51949: "euc_kr",
    52936: "hz",
    65000: "utf_7",
    65001: "utf_8_sig",
}


def get_encoding(code_page: int) -> Optional[str]:
    return _ENCODINGS.get(code_page)


def get_entry(archive: ZipFile, entry_name: str) -> Optional[ZipInfo]:
    return next((zi for zi in archive.filelist if zi.filename == entry_name), None)


def find_entry(archive: ZipFile, predicate: Callable[[ZipInfo], bool]) -> Optional[ZipInfo]:
    return next((zi for zi in archive.filelist if predicate(zi)), None)
