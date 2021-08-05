import os
import platform
from glob import glob
from typing import Generator, Iterable, Tuple, TypeVar, cast

import regex

from ..scripture.canon import book_id_to_number


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
            converted_mask = cast(str, regex.escape(base)).replace("\\*", "(.*)").replace("\\?", "(.)")
            mask_regex = regex.compile(converted_mask, regex.IGNORECASE if platform.system() == "Windows" else 0)

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


def merge_verse_ranges(verse1: str, verse2: str) -> str:
    pass
