from typing import Generator, List, Union

from machine.corpora import (
    QuotationDenormalizationAction,
    QuotationDenormalizationSettings,
    QuotationDenormalizationUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmParserHandler,
    UsfmToken,
    UsfmTokenType,
    UsfmUpdateBlock,
    UsfmUpdateBlockElement,
    UsfmUpdateBlockElementType,
    parse_usfm,
)
from machine.corpora.analysis import (
    QuotationMarkDirection,
    QuotationMarkFinder,
    QuotationMarkMetadata,
    QuotationMarkResolutionIssue,
    QuotationMarkResolutionSettings,
    QuotationMarkResolver,
    QuotationMarkStringMatch,
    QuoteConventionSet,
    TextSegment,
    UsfmMarkerType,
    standard_quote_conventions,
)

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


def test_quotes_spanning_verses() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    \\v 2 'You shall not eat of any tree of the garden'?"
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, \n"
        + "\\v 2 ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_single_embed() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    \\f + \\ft "This is a 'footnote'" \\f*
    of the field which Yahweh God had made.
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal "
        + "\\f + \\ft “This is a ‘footnote’” \\f* of the field which Yahweh God had made."
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_multiple_embeds() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    \\f + \\ft "This is a 'footnote'" \\f*
    of the field \\f + \\ft Second "footnote" here \\f* which Yahweh God had made.
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal "
        + "\\f + \\ft “This is a ‘footnote’” \\f* of the field \\f + \\ft Second "
        + "“footnote” here \\f* which Yahweh God had made."
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_quotes_in_text_and_embed() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really \\f + \\ft a
    "footnote" in the "midst of 'text'" \\f* said,
    'You shall not eat of any tree of the garden'?"
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really \\f + \\ft a “footnote” in the “midst of ‘text’” \\f* "
        + "said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_quotes_in_multiple_verses_and_embed() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God
    \\v 2 really \\f + \\ft a
    "footnote" in the "midst of 'text'" \\f* said,
    'You shall not eat of any tree of the garden'?"
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God\n"
        + "\\v 2 really \\f + \\ft a “footnote” in the “midst of ‘text’” \\f* "
        + "said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# Basic denormalization does not consider the nesting of quotation marks,
# but only determines opening/closing marks and maps based on that.
def test_basic_quotation_denormalization_same_as_full() -> None:
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
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_BASIC),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_basic_quotation_denormalization_incorrectly_nested() -> None:
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
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_BASIC),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_basic_quotation_denormalization_incorrectly_nested_second_case() -> None:
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
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_BASIC),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_basic_quotation_denormalization_unclosed_quote() -> None:
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
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_BASIC),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_default_denormalization_action() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_skipped_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_FULL),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.APPLY_BASIC),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(default_chapter_action=QuotationDenormalizationAction.SKIP),
    )
    assert_usfm_equal(observed_usfm, expected_skipped_usfm)


def test_single_chapter_denormalization_action() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_skipped_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(chapter_actions=[QuotationDenormalizationAction.APPLY_FULL]),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(chapter_actions=[QuotationDenormalizationAction.APPLY_BASIC]),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(chapter_actions=[QuotationDenormalizationAction.SKIP]),
    )
    assert_usfm_equal(observed_usfm, expected_skipped_usfm)


def test_multiple_chapter_same_denormalization_action() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle" than any animal
    of the field which Yahweh God had made.
    \\c 2
    \\v 1 He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + '\\v 1 Now the serpent was more subtle" than any animal of the field which Yahweh God had made.\n'
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(
            chapter_actions=[QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_FULL]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(
            chapter_actions=[QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_BASIC]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)


def test_multiple_chapter_multiple_denormalization_actions() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle" than any animal
    of the field which Yahweh God had made.
    \\c 2
    \\v 1 He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_then_basic_usfm = (
        "\\c 1\n"
        + '\\v 1 Now the serpent was more subtle" than any animal of the field which Yahweh God had made.\n'
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_basic_then_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_then_skip_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + '\\v 1 He said to the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(
            chapter_actions=[QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_BASIC]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_full_then_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(
            chapter_actions=[QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_FULL]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_basic_then_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings(
            chapter_actions=[QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.SKIP]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_basic_then_skip_usfm)


def test_process_scripture_element() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "british_english")
    )
    quotation_denormalizer._quotation_mark_finder = MockQuotationMarkFinder()

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[UsfmToken(UsfmTokenType.TEXT, text="test segment")],
    )
    mock_quotation_mark_resolver: QuotationMarkResolver = MockQuotationMarkResolver()
    quotation_denormalizer._process_scripture_element(update_element, mock_quotation_mark_resolver)

    assert quotation_denormalizer._quotation_mark_finder.num_times_called == 1
    assert mock_quotation_mark_resolver.num_times_called == 1
    assert quotation_denormalizer._quotation_mark_finder.matches_to_return[0].text_segment.text == "this is a ‘test"
    assert quotation_denormalizer._quotation_mark_finder.matches_to_return[1].text_segment.text == "the test ends” here"


def test_create_text_segments_basic() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT, tokens=[UsfmToken(UsfmTokenType.TEXT, text="test segment")]
    )
    text_segments: List[TextSegment] = quotation_denormalizer._create_text_segments(update_element)

    assert len(text_segments) == 1
    assert text_segments[0].text == "test segment"
    assert text_segments[0].immediate_preceding_marker is UsfmMarkerType.NoMarker
    assert text_segments[0].markers_in_preceding_context == set()
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment is None


def test_create_text_segments_with_preceding_markers() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.PARAGRAPH),
            UsfmToken(UsfmTokenType.TEXT, text="test segment"),
        ],
    )
    text_segments: List[TextSegment] = quotation_denormalizer._create_text_segments(update_element)

    assert len(text_segments) == 1
    assert text_segments[0].text == "test segment"
    assert text_segments[0].immediate_preceding_marker == UsfmMarkerType.ParagraphMarker
    assert text_segments[0].markers_in_preceding_context == {
        UsfmMarkerType.VerseMarker,
        UsfmMarkerType.ParagraphMarker,
    }
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment is None


def test_create_text_segments_with_multiple_text_tokens() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.PARAGRAPH),
            UsfmToken(UsfmTokenType.TEXT, text="test segment1"),
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.CHARACTER),
            UsfmToken(UsfmTokenType.TEXT, text="test segment2"),
            UsfmToken(UsfmTokenType.PARAGRAPH),
        ],
    )
    text_segments: List[TextSegment] = quotation_denormalizer._create_text_segments(update_element)

    assert len(text_segments) == 2
    assert text_segments[0].text == "test segment1"
    assert text_segments[0].immediate_preceding_marker == UsfmMarkerType.ParagraphMarker
    assert text_segments[0].markers_in_preceding_context == {UsfmMarkerType.VerseMarker, UsfmMarkerType.ParagraphMarker}
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment == text_segments[1]
    assert text_segments[1].text == "test segment2"
    assert text_segments[1].immediate_preceding_marker == UsfmMarkerType.CharacterMarker
    assert text_segments[1].markers_in_preceding_context == {UsfmMarkerType.VerseMarker, UsfmMarkerType.CharacterMarker}
    assert text_segments[1].previous_segment == text_segments[0]
    assert text_segments[1].next_segment is None


def test_create_text_segment() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    usfm_token: UsfmToken = UsfmToken(UsfmTokenType.TEXT, text="test segment")
    segment: Union[TextSegment, None] = quotation_denormalizer._create_text_segment(usfm_token)

    assert segment is not None
    assert segment.text == "test segment"
    assert segment.immediate_preceding_marker is UsfmMarkerType.NoMarker
    assert segment.markers_in_preceding_context == set()
    assert segment.usfm_token == usfm_token


def test_set_previous_and_next_for_segments() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    segments: List[TextSegment] = [
        TextSegment.Builder().set_text("segment 1 text").build(),
        TextSegment.Builder().set_text("segment 2 text").build(),
        TextSegment.Builder().set_text("segment 3 text").build(),
    ]

    quotation_denormalizer._set_previous_and_next_for_segments(segments)

    assert segments[0].previous_segment is None
    assert segments[0].next_segment == segments[1]
    assert segments[1].previous_segment == segments[0]
    assert segments[1].next_segment == segments[2]
    assert segments[2].previous_segment == segments[1]
    assert segments[2].next_segment is None


def test_check_for_chapter_change() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler("standard_english", "standard_english")
    )

    assert quotation_denormalizer._current_chapter_number == 0

    quotation_denormalizer._check_for_chapter_change(UsfmUpdateBlock([ScriptureRef.parse("MAT 1:1")], []))

    assert quotation_denormalizer._current_chapter_number == 1

    quotation_denormalizer._check_for_chapter_change(UsfmUpdateBlock([ScriptureRef.parse("ISA 15:22")], []))

    assert quotation_denormalizer._current_chapter_number == 15


def test_start_new_chapter() -> None:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
        create_quotation_denormalization_usfm_update_block_handler(
            "standard_english",
            "standard_english",
            QuotationDenormalizationSettings(
                chapter_actions=[
                    QuotationDenormalizationAction.SKIP,
                    QuotationDenormalizationAction.APPLY_FULL,
                    QuotationDenormalizationAction.APPLY_BASIC,
                ]
            ),
        )
    )

    quotation_denormalizer._next_scripture_text_segment_builder.add_preceding_marker(
        UsfmMarkerType.EmbedMarker
    ).set_text("this text should be erased")
    quotation_denormalizer._verse_text_quotation_mark_resolver._issues.add(
        QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK
    )

    quotation_denormalizer._start_new_chapter(1)
    segment = quotation_denormalizer._next_scripture_text_segment_builder.build()
    assert quotation_denormalizer._current_denormalization_action == QuotationDenormalizationAction.SKIP
    assert segment.immediate_preceding_marker == UsfmMarkerType.ChapterMarker
    assert segment.text == ""
    assert UsfmMarkerType.EmbedMarker not in segment.markers_in_preceding_context
    assert quotation_denormalizer._verse_text_quotation_mark_resolver._issues == set()

    quotation_denormalizer._start_new_chapter(2)
    assert quotation_denormalizer._current_denormalization_action == QuotationDenormalizationAction.APPLY_FULL

    quotation_denormalizer._start_new_chapter(3)
    assert quotation_denormalizer._current_denormalization_action == QuotationDenormalizationAction.APPLY_BASIC


def denormalize_quotation_marks(
    normalized_usfm: str,
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_denormalization_settings: QuotationDenormalizationSettings = QuotationDenormalizationSettings(),
) -> str:
    quotation_denormalizer: QuotationDenormalizationUsfmUpdateBlockHandler = (
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
    quotation_denormalization_settings: QuotationDenormalizationSettings = QuotationDenormalizationSettings(),
) -> QuotationDenormalizationUsfmUpdateBlockHandler:
    source_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    return QuotationDenormalizationUsfmUpdateBlockHandler(
        source_quote_convention,
        target_quote_convention,
        quotation_denormalization_settings,
    )


def assert_usfm_equal(observed_usfm: str, expected_usfm: str) -> None:
    for observed_line, expected_line in zip(observed_usfm.split("\n"), expected_usfm.split("\n")):
        assert observed_line.strip() == expected_line.strip()


class MockQuotationMarkFinder(QuotationMarkFinder):
    def __init__(self) -> None:
        super().__init__(QuoteConventionSet([]))
        self.num_times_called = 0
        self.matches_to_return = [
            QuotationMarkStringMatch(TextSegment.Builder().set_text('this is a "test').build(), 10, 11),
            QuotationMarkStringMatch(TextSegment.Builder().set_text('the test ends" here').build(), 13, 14),
        ]

    def find_all_potential_quotation_marks_in_text_segments(
        self, text_segments: List[TextSegment]
    ) -> List[QuotationMarkStringMatch]:
        self.num_times_called += 1
        return self.matches_to_return


class MockQuotationMarkResolver(QuotationMarkResolver):
    def __init__(self):
        super().__init__(QuotationMarkResolutionSettings())
        self.num_times_called = 0

    def resolve_quotation_marks(
        self, quote_matches: List[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        self.num_times_called += 1
        current_depth = 1
        current_direction = QuotationMarkDirection.Opening
        for quote_match in quote_matches:
            yield quote_match.resolve(current_depth, current_direction)
            current_depth += 1
            current_direction = (
                QuotationMarkDirection.Closing
                if current_direction == QuotationMarkDirection.Opening
                else QuotationMarkDirection.Opening
            )
