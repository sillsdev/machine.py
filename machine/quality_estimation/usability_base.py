from .usability_label import UsabilityLabel


class UsabilityBase:
    def __init__(self, label: UsabilityLabel, projected_chrf3: float, usability: float, confidence: float) -> None:
        self.confidence = confidence
        self.label = label
        self.projected_chrf3 = projected_chrf3
        self.usability = usability
