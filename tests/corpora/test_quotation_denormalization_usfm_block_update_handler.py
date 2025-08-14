from typing import Union

from machine.corpora import (
    QuotationMarkDenormalizationUsfmUpdateBlockHandler,
    QuotationMarkUpdateSettings,
    QuotationMarkUpdateStrategy,
    UpdateUsfmParserHandler,
    parse_usfm,
)
from machine.punctuation_analysis import STANDARD_QUOTE_CONVENTIONS, QuoteConvention

simple_normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?"
    """


def test_simple_english_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_british_english_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, 'Has God really said,
    "You shall not eat of any tree of the garden"?'
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ‘Has God really said, “You shall not eat of any tree of the garden”?’"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "british_english", "british_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# no denormalization should be needed for this example
def test_simple_typewriter_english_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, \"Has God really said, 'You shall not eat of any tree of the garden'?\""
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "typewriter_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# some of the quotes shouldn't need to be denormalized
def test_simple_hybrid_typewriter_english_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, 'You shall not eat of any tree of the garden'?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "hybrid_typewriter_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# the single guillemets shouldn't need to be denormalized
# because Moses doesn't normalize them
def test_simple_french_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    ‹You shall not eat of any tree of the garden›?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, «Has God really said, ‹You shall not eat of any tree of the garden›?»"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_french", "standard_french")
    assert_usfm_equal(observed_usfm, expected_usfm)


# the unusual quotation marks shouldn't need to be denormalized
def test_simple_typewriter_french_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, <<Has God really said,
    <You shall not eat of any tree of the garden>?>>
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, <<Has God really said, <You shall not eat of any tree of the garden>?>>"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "typewriter_french", "typewriter_french")
    assert_usfm_equal(observed_usfm, expected_usfm)


# the 1st- and 2nd-level quotes are denormalized to identical marks
def test_simple_western_european_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden"?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, «Has God really said, “You shall not eat of any tree of the garden”?»"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "western_european", "western_european")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_typewriter_western_european_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, <<Has God really said,
    "You shall not eat of any tree of the garden"?>>
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, <<Has God really said, "You shall not eat of any tree of the garden"?>>'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm, "typewriter_western_european", "typewriter_western_european"
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_typewriter_western_european_variant_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    <You shall not eat of any tree of the garden>?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, "Has God really said, <You shall not eat of any tree of the garden>?"'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm, "typewriter_western_european_variant", "typewriter_western_european_variant"
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_hybrid_typewriter_western_european_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden"?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, «Has God really said, "You shall not eat of any tree of the garden"?»'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm, "hybrid_typewriter_western_european", "hybrid_typewriter_western_european"
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_central_european_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden"?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, „Has God really said, ‚You shall not eat of any tree of the garden‘?“"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "central_european", "central_european")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_central_european_guillemets_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    ›You shall not eat of any tree of the garden‹?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, »Has God really said, ›You shall not eat of any tree of the garden‹?«"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm, "central_european_guillemets", "central_european_guillemets"
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_swedish_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ”Has God really said, ’You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_swedish", "standard_swedish")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_finnish_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, »Has God really said, ’You shall not eat of any tree of the garden’?»"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_finnish")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_eastern_european_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, „Has God really said, ‚You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "eastern_european")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_russian_quote_denormalization() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden"?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, «Has God really said, „You shall not eat of any tree of the garden“?»"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_russian", "standard_russian")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_arabic_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ”Has God really said, ’You shall not eat of any tree of the garden‘?“"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_arabic")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_quotation_denormalization_same_as_full() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_quotation_denormalization_incorrectly_nested() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden"?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, “You shall not eat of any tree of the garden”?”"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_quotation_denormalization_incorrectly_nested_second_case() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, 'Has God really said,
    "You shall not eat of any tree of the garden"?'
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ‘Has God really said, “You shall not eat of any tree of the garden”?’"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_quotation_denormalization_unclosed_quote() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def denormalize_quotation_marks(
    normalized_usfm: str,
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_denormalization_settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
) -> str:
    quotation_denormalizer: QuotationMarkDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler(
            source_quote_convention_name, target_quote_convention_name, quotation_denormalization_settings
        )
    )

    updater = UpdateUsfmParserHandler(update_block_handlers=[quotation_denormalizer])
    parse_usfm(normalized_usfm, updater)

    return updater.get_usfm()


def create_quotation_denormalization_usfm_update_block_handler(
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_denormalization_settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
) -> QuotationMarkDenormalizationUsfmUpdateBlockHandler:
    source_quote_convention = get_quote_convention_by_name(source_quote_convention_name)
    target_quote_convention = get_quote_convention_by_name(target_quote_convention_name)

    return QuotationMarkDenormalizationUsfmUpdateBlockHandler(
        source_quote_convention,
        target_quote_convention,
        quotation_denormalization_settings,
    )


def assert_usfm_equal(observed_usfm: str, expected_usfm: str) -> None:
    for observed_line, expected_line in zip(observed_usfm.split("\n"), expected_usfm.split("\n")):
        assert observed_line.strip() == expected_line.strip()


def get_quote_convention_by_name(name: str) -> QuoteConvention:
    quote_convention: Union[QuoteConvention, None] = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(name)
    assert quote_convention is not None
    return quote_convention
