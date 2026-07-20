from ..corpora import ScriptureRef
from .score import Score


class ScriptureSegmentScore(Score):
    def __init__(self, slope: float, confidence: float, intercept: float, scripture_ref: ScriptureRef) -> None:
        super().__init__(slope, confidence, intercept)
        self.scripture_ref = scripture_ref
