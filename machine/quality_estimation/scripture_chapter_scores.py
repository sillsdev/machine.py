from collections import defaultdict

from .score import Score


class ScriptureChapterScores:
    def __init__(self) -> None:
        self._segment_usabilities: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.scores: dict[str, dict[int, Score]] = defaultdict(dict)

    def add_score(self, book: str, chapter: int, score: Score) -> None:
        self.scores[book][chapter] = score

    def get_score(self, book: str, chapter: int) -> Score | None:
        return self.scores.get(book, {}).get(chapter, None)

    def append_segment_usability(self, book: str, chapter: int, usability: float) -> None:
        self._segment_usabilities[book][chapter].append(usability)

    def get_segment_usabilities(self, book: str, chapter: int) -> list[float]:
        return list(self._segment_usabilities.get(book, {}).get(chapter, []))
