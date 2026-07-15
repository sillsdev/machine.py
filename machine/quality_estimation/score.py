class Score:
    def __init__(self, slope: float, confidence: float, intercept: float) -> None:
        self.confidence = confidence
        self.projected_chrf3 = slope * confidence + intercept
