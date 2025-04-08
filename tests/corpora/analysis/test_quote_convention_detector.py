from typing import Union

from machine.corpora import parse_usfm
from machine.corpora.analysis import QuoteConventionAnalysis, QuoteConventionDetector

# Text comes from the World English Bible, which is in the public domain.


def test_standard_english() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?”
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_english"


def test_typewriter_english() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, \"Has God really said,
    'You shall not eat of any tree of the garden'?\"
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "typewriter_english"


def test_british_english() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ‘Has God really said,
    “You shall not eat of any tree of the garden”?’
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "british_english"


def test_british_typewriter_english() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, 'Has God really said,
    \"You shall not eat of any tree of the garden\"?'
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "british_typewriter_english"


def test_hybrid_typewriter_english() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    'You shall not eat of any tree of the garden'?”
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "hybrid_typewriter_english"


def test_standard_french() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    ‹You shall not eat of any tree of the garden›?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_french"


def test_typewriter_french() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, <<Has God really said,
    <You shall not eat of any tree of the garden>?>>
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "typewriter_french"


# french_variant requires a 3rd-level of quotes to differentiate from standard_french


def test_western_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    “You shall not eat of any tree of the garden”?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "western_european"


def test_british_inspired_western_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    ‘You shall not eat of any tree of the garden’?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "british_inspired_western_european"


def test_typewriter_western_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, <<Has God really said,
    "You shall not eat of any tree of the garden"?>>
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "typewriter_western_european"


def test_typewriter_western_european_variant() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    <You shall not eat of any tree of the garden>?"
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "typewriter_western_european_variant"


def test_hybrid_typewriter_western_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    "You shall not eat of any tree of the garden"?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "hybrid_typewriter_western_european"


def test_hybrid_british_typewriter_western_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    'You shall not eat of any tree of the garden'?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "hybrid_british_typewriter_western_european"


def test_central_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, „Has God really said,
    ‚You shall not eat of any tree of the garden‘?“
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "central_european"


def test_central_european_guillemets() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, »Has God really said,
    ›You shall not eat of any tree of the garden‹?«
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "central_european_guillemets"


def test_standard_swedish() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ”Has God really said,
    ’You shall not eat of any tree of the garden’?”
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_swedish"


def test_standard_finnish() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, »Has God really said,
    ’You shall not eat of any tree of the garden’?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_finnish"


def test_eastern_european() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, „Has God really said,
    ‚You shall not eat of any tree of the garden’?”
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "eastern_european"


def test_standard_russian() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    „You shall not eat of any tree of the garden“?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_russian"


def test_standard_arabic() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ”Has God really said,
    ’You shall not eat of any tree of the garden‘?“
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_arabic"


def test_non_standard_arabic() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    ’You shall not eat of any tree of the garden‘?»
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "non-standard_arabic"


def test_mismatched_quotation_marks() -> None:
    usfm = """
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?”
    \\v 2 The woman said to the serpent, 
    “We may eat fruit from the trees of the garden, 
    \\v 3 but not the fruit of the tree which is in the middle of the garden. 
    God has said, ‘You shall not eat of it. You shall not touch it, lest you die.’
    """
    analysis = detect_quote_convention(usfm)
    assert analysis is not None
    assert analysis.get_best_quote_convention().get_name() == "standard_english"


def detect_quote_convention(usfm: str) -> Union[QuoteConventionAnalysis, None]:
    quote_convention_detector = QuoteConventionDetector()
    parse_usfm(usfm, quote_convention_detector)
    return quote_convention_detector.detect_quotation_convention(print_summary=False)
