from pytest import raises

from machine.quality_estimation import LOW_BOOK_CONFIDENCE_THRESHOLD, is_book_confidence_unusually_low


def test_is_book_confidence_unusually_low_at_threshold() -> None:
    assert not is_book_confidence_unusually_low(LOW_BOOK_CONFIDENCE_THRESHOLD)


def test_is_book_confidence_unusually_low_min_confidence() -> None:
    assert is_book_confidence_unusually_low(0.0)


def test_is_book_confidence_unusually_low_max_confidence() -> None:
    assert not is_book_confidence_unusually_low(1.0)


def test_is_book_confidence_unusually_low_with_book_id_and_model() -> None:
    assert is_book_confidence_unusually_low(0.3, "MAT", "facebook/nllb-200-distilled-1.3B")
    assert not is_book_confidence_unusually_low(0.9, "MAT", "facebook/nllb-200-distilled-1.3B")


def test_is_book_confidence_unusually_low_negative() -> None:
    with raises(ValueError):
        is_book_confidence_unusually_low(-0.5)


def test_is_book_confidence_unusually_low_greater_than_one() -> None:
    with raises(ValueError):
        is_book_confidence_unusually_low(1.5)


def test_is_book_confidence_unusually_low_nan() -> None:
    with raises(ValueError):
        is_book_confidence_unusually_low(float("nan"))
