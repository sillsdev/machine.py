from dataclasses import dataclass

from .verse import Verse


@dataclass(frozen=True)
class Chapter:
    verses: list[Verse]
    chatper_num: int = 0
