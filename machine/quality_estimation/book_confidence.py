LOW_BOOK_CONFIDENCE_THRESHOLD = 0.42


def is_book_confidence_unusually_low(confidence: float) -> bool:
    if not 0 <= confidence <= 1:
        raise ValueError(
            f"The book confidence {confidence} is invalid. "
            f"It is calculated as the geometric mean of the segment confidences, "
            f"and it must be between 0 and 1, inclusive."
        )
    return confidence < LOW_BOOK_CONFIDENCE_THRESHOLD
