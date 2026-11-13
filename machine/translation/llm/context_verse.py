from dataclasses import dataclass


@dataclass(frozen=True)
class ContextVerse:
    offset_from_segment: int
    source: str
    target: str = ""
