from testutils.corpora_test_helpers import ignore_line_endings

from machine.corpora import UpdateUsfmParserHandler, parse_usfm
from machine.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuotationMarkDenormalizationFirstPass,
    QuotationMarkDenormalizationUsfmUpdateBlockHandler,
    QuotationMarkUpdateSettings,
)


def test_full_quotation_denormalization_pipeline() -> None:
    normalized_usfm = """
    \\id GEN
    \\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?"
    \\v 2 The woman said to the serpent,
    "We may eat fruit from the trees of the garden,
    \\v 3 but not the fruit of the tree which is in the middle of the garden.
    God has said, 'You shall not eat of it. You shall not touch it, lest you die.'"
    """

    expected_denormalized_usfm = """\\id GEN
\\c 1
\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to the woman, “Has God really said, ‘You shall not eat of any tree of the garden’?”
\\v 2 The woman said to the serpent, “We may eat fruit from the trees of the garden,
\\v 3 but not the fruit of the tree which is in the middle of the garden. God has said, ‘You shall not eat of it. You shall not touch it, lest you die.’”
"""  # noqa: E501

    standard_english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert standard_english_quote_convention is not None

    quotation_mark_denormalization_first_pass = QuotationMarkDenormalizationFirstPass(standard_english_quote_convention)

    parse_usfm(normalized_usfm, quotation_mark_denormalization_first_pass)
    best_chapter_strategies = quotation_mark_denormalization_first_pass.find_best_chapter_strategies()

    assert [chapter for chapter, _ in best_chapter_strategies] == [1]

    quotation_mark_denormalizer = QuotationMarkDenormalizationUsfmUpdateBlockHandler(
        standard_english_quote_convention,
        QuotationMarkUpdateSettings(chapter_strategies=[strategy for _, strategy in best_chapter_strategies]),
    )

    updater = UpdateUsfmParserHandler(update_block_handlers=[quotation_mark_denormalizer])
    parse_usfm(normalized_usfm, updater)

    actual_denormalized_usfm = updater.get_usfm()

    ignore_line_endings(actual_denormalized_usfm, expected_denormalized_usfm)
