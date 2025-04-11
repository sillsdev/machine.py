from machine.corpora import QuotationDenormalizationScriptureUpdateBlockHandler, UpdateUsfmParserHandler, parse_usfm
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_english")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "british_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# no denormalization should be needed for this example
def test_simple_typewriter_english_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, \"Has God really said, 'You shall not eat of any tree of the garden'?\""
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "typewriter_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# some of the quotes shouldn't need to be denormalized
def test_simple_hybrid_typewriter_english_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, 'You shall not eat of any tree of the garden'?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "hybrid_typewriter_english")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_french")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "typewriter_french")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "western_european")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "typewriter_western_european")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "typewriter_western_european_variant")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "hybrid_typewriter_western_european")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "central_european")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "central_european_guillemets")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_swedish")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_finnish_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, »Has God really said, ’You shall not eat of any tree of the garden’?»"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_finnish")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_eastern_european_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, „Has God really said, ‚You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "eastern_european")
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

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_russian")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_simple_arabic_quote_denormalization() -> None:
    normalized_usfm = simple_normalized_usfm
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ”Has God really said, ’You shall not eat of any tree of the garden‘?“"
    )

    observed_usfm = denormalize_quotation_marks(normalized_usfm, "standard_arabic")
    assert_usfm_equal(observed_usfm, expected_usfm)


def denormalize_quotation_marks(normalized_usfm: str, quote_convention_name: str) -> str:
    standard_english_quote_convention = (
        standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(quote_convention_name)
    )
    assert standard_english_quote_convention is not None

    quotation_denormalizer: QuotationDenormalizationScriptureUpdateBlockHandler = (
        QuotationDenormalizationScriptureUpdateBlockHandler(standard_english_quote_convention)
    )
    updater = UpdateUsfmParserHandler(update_block_handlers=[quotation_denormalizer])
    parse_usfm(normalized_usfm, updater)

    return updater.get_usfm()


def assert_usfm_equal(observed_usfm: str, expected_usfm: str) -> None:
    for observed_line, expected_line in zip(observed_usfm.split("\n"), expected_usfm.split("\n")):
        assert observed_line.strip() == expected_line.strip()
