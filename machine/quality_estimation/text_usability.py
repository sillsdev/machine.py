from .usability_base import UsabilityBase
from .usability_label import UsabilityLabel


class TextUsability(UsabilityBase):
    def __init__(
        self, text_id: str, label: UsabilityLabel, projected_chrf3: float, usability: float, confidence: float
    ) -> None:
        super().__init__(label, projected_chrf3, usability, confidence)
        self.text_id = text_id
