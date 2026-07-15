from ..corpora import MultiKeyRef
from .score import Score


class TextSegmentScore(Score):
    def __init__(self, slope: float, confidence: float, intercept: float, segment_ref: MultiKeyRef) -> None:
        super().__init__(slope, confidence, intercept)
        self.segment_ref = segment_ref
        self.text_id = segment_ref.text_id
