from ..corpora import MultiKeyRef
from .text_usability import TextUsability
from .usability_label import UsabilityLabel


class TextSegmentUsability(TextUsability):
    def __init__(
        self,
        segment_ref: MultiKeyRef,
        label: UsabilityLabel,
        projected_chrf3: float,
        usability: float,
        confidence: float,
    ) -> None:
        super().__init__(segment_ref.text_id, label, projected_chrf3, usability, confidence)
        self.segment_ref = segment_ref
