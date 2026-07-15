from ..corpora import ScriptureRef
from .scripture_chapter_usability import ScriptureChapterUsability
from .usability_label import UsabilityLabel


class ScriptureSegmentUsability(ScriptureChapterUsability):
    def __init__(
        self,
        scripture_ref: ScriptureRef,
        label: UsabilityLabel,
        projected_chrf3: float,
        usability: float,
        confidence: float,
    ) -> None:
        super().__init__(scripture_ref.book, scripture_ref.chapter_num, label, projected_chrf3, usability, confidence)
        self.scripture_ref = scripture_ref
