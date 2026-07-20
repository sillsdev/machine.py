from collections import defaultdict

from .score import Score


class TextScores:
    def __init__(self) -> None:
        self._segment_usabilities: dict[str, list[float]] = defaultdict(list)
        self.scores: dict[str, Score] = {}

    def add_score(self, target_draft_file_stem: str, score: Score) -> None:
        self.scores[target_draft_file_stem] = score

    def get_score(self, target_draft_file_stem: str) -> Score | None:
        return self.scores.get(target_draft_file_stem, None)

    def append_segment_usability(self, text_id: str, usability: float) -> None:
        self._segment_usabilities[text_id].append(usability)

    def get_segment_usabilities(self, text_id: str) -> list[float]:
        return list(self._segment_usabilities.get(text_id, []))
