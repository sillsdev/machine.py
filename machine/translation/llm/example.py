from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    source: str
    target: str
