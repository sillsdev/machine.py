from .scripture_book_usability import ScriptureBookUsability
from .usability_label import UsabilityLabel


class ScriptureChapterUsability(ScriptureBookUsability):
    def __init__(
        self,
        book: str,
        chapter: int,
        label: UsabilityLabel,
        projected_chrf3: float,
        usability: float,
        confidence: float,
    ) -> None:
        super().__init__(book, label, projected_chrf3, usability, confidence)
        self.chapter = chapter
