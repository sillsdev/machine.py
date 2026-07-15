from .usability_label import UsabilityLabel


class Thresholds:
    def __init__(self, green_threshold: float, yellow_threshold: float) -> None:
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold

    def return_label(self, probability: float) -> UsabilityLabel:
        if probability >= self.green_threshold:
            return UsabilityLabel.GREEN
        elif probability >= self.yellow_threshold:
            return UsabilityLabel.YELLOW
        else:
            return UsabilityLabel.RED
