from .usability_base import UsabilityBase
from .usability_label import UsabilityLabel


class ScriptureBookUsability(UsabilityBase):
    def __init__(
        self, book: str, label: UsabilityLabel, projected_chrf3: float, usability: float, confidence: float
    ) -> None:
        super().__init__(label, projected_chrf3, usability, confidence)
        self.book = book
