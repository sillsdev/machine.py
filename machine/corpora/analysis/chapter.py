from dataclasses import dataclass

from .verse import Verse


@dataclass
class Chapter:
    verses: list[Verse]
