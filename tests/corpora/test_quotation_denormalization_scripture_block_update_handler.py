from machine.corpora import (
    QuotationDenormalizationAction,
    QuotationDenormalizationScriptureUpdateBlockHandler,
    QuotationDenormalizationSettings,
    UpdateUsfmParserHandler,
    parse_usfm,
)
from machine.corpora.analysis import standard_quote_conventions

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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_BASIC)
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_BASIC)
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_BASIC)
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_BASIC)
        .build(),
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
        QuotationDenormalizationSettings.Builder().run_on_existing_text().build(),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_FULL)
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.APPLY_BASIC)
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_default_chapter_action(QuotationDenormalizationAction.SKIP)
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_FULL])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_BASIC])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.SKIP])
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_FULL])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_BASIC])
        .build(),
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
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_BASIC])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_full_then_basic_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_FULL])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_basic_then_full_usfm)

    observed_usfm = denormalize_quotation_marks(
        normalized_usfm,
        "standard_english",
        "standard_english",
        QuotationDenormalizationSettings.Builder()
        .run_on_existing_text()
        .set_chapter_actions([QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.SKIP])
        .build(),
    )
    assert_usfm_equal(observed_usfm, expected_basic_then_skip_usfm)


def denormalize_quotation_marks(
    normalized_usfm: str,
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_denormalization_settings: QuotationDenormalizationSettings = QuotationDenormalizationSettings.Builder()
    .run_on_existing_text()
    .build(),
) -> str:
    source_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    quotation_denormalizer: QuotationDenormalizationScriptureUpdateBlockHandler = (
        QuotationDenormalizationScriptureUpdateBlockHandler(
            source_quote_convention,
            target_quote_convention,
            quotation_denormalization_settings,
        )
    )

    updater = UpdateUsfmParserHandler(update_block_handlers=[quotation_denormalizer])
    parse_usfm(normalized_usfm, updater)

    return updater.get_usfm()


def assert_usfm_equal(observed_usfm: str, expected_usfm: str) -> None:
    for observed_line, expected_line in zip(observed_usfm.split("\n"), expected_usfm.split("\n")):
        assert observed_line.strip() == expected_line.strip()
