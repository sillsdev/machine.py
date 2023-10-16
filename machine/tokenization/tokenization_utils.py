from typing import Generator, Iterable, List

from ..annotations.range import Range


def split(s: str, ranges: Iterable[Range[int]]) -> List[str]:
    return [s[range.start : range.end] for range in ranges]


def get_ranges(s: str, tokens: Iterable[str]) -> Generator[Range[int], None, None]:
    start = 0
    for token in tokens:
        index = s.find(token, start)
        if index == -1:
            raise ValueError(f"The string does not contain the specified token: {token}.")
        yield Range.create(index, index + len(token))
        start = index + len(token)
