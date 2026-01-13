from dataclasses import dataclass
from typing import List

from ..scripture.verse_ref import VerseRef


@dataclass(frozen=True)
class KeyTerm:
    id: str
    category: str
    domain: str
    renderings: List[str]
    references: List[VerseRef]
    renderings_patterns: List[str]
