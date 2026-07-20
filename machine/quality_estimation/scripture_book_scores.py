from collections import defaultdict

from .score import Score


class ScriptureBookScores:
    def __init__(self) -> None:
        self._segment_usabilities: dict[str, list[float]] = defaultdict(list)
        self.scores: dict[str, Score] = {}

    def add_score(self, book: str, score: Score) -> None:
        self.scores[book] = score

    def get_score(self, book: str) -> Score | None:
        return self.scores.get(book, None)

    def append_segment_usability(self, book: str, usability: float) -> None:
        self._segment_usabilities[book].append(usability)

    def get_segment_usabilities(self, book: str) -> list[float]:
        return list(self._segment_usabilities.get(book, []))
