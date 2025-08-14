from dataclasses import dataclass

from .verse import Verse


@dataclass(frozen=True)
class Chapter:
    verses: list[Verse]
