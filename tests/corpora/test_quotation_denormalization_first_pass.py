from typing import List, Union

from machine.corpora import QuotationDenormalizationAction, QuotationDenormalizationFirstPass, parse_usfm
from machine.corpora.analysis import (
    Chapter,
    QuotationMarkResolutionIssue,
    QuoteConvention,
    TextSegment,
    Verse,
    standard_quote_conventions,
)


def test_check_whether_basic_denormalization_will_work() -> None:

    first_pass_analyzer = QuotationDenormalizationFirstPass(QuoteConvention("", []), QuoteConvention("", []))

    # Cases where we expect basic denormalization to work
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_english"),
            get_quote_convention_by_name("standard_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_french"),
            get_quote_convention_by_name("british_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("typewriter_western_european"),
            get_quote_convention_by_name("standard_russian"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("typewriter_western_european_variant"),
            get_quote_convention_by_name("standard_arabic"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("central_european"),
            get_quote_convention_by_name("british_typewriter_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_swedish"),
            get_quote_convention_by_name("typewriter_french"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_finnish"),
            get_quote_convention_by_name("british_inspired_western_european"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("eastern_european"),
            get_quote_convention_by_name("central_european"),
        )
        is True
    )

    # Cases where we expect basic denormalization to fail
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("western_european"),
            get_quote_convention_by_name("standard_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("french_variant"),
            get_quote_convention_by_name("hybrid_typewriter_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("british_inspired_western_european"),
            get_quote_convention_by_name("standard_russian"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("typewriter_english"),
            get_quote_convention_by_name("western_european"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("central_european_guillemets"),
            get_quote_convention_by_name("french_variant"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_arabic"),
            get_quote_convention_by_name("hybrid_typewriter_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_basic_denormalization_will_work(
            get_quote_convention_by_name("standard_russian"),
            get_quote_convention_by_name("standard_french"),
        )
        is False
    )


def test_choose_best_action_for_chapter() -> None:
    # Verse text with no issues
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman, "Has God really said, '
            + "'You shall not eat of any tree of the garden'?\""
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.APPLY_FULL
    assert actual_action == expected_action

    # Verse text with unpaired opening quotation mark
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman, "Has God really said, '
            + "'You shall not eat of any tree of the garden'?"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.APPLY_BASIC
    assert actual_action == expected_action

    # Verse text with unpaired closing quotation mark
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + "He said to the woman, Has God really said, "
            + 'You shall not eat of any tree of the garden?"'
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.APPLY_BASIC
    assert actual_action == expected_action

    # Verse text with too deeply nested quotation marks
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            '"Now the serpent was more "subtle than any animal '
            + 'of the "field which "Yahweh God had made. '
            + 'He said to the woman, "Has God really said, '
            + '"You shall not eat of any tree of the garden?'
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.SKIP
    assert actual_action == expected_action

    # Verse text with an ambiguous quotation mark
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman"Has God really said, '
            + "You shall not eat of any tree of the garden?"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.SKIP
    assert actual_action == expected_action

    # Verse text with an ambiguous quotation mark
    actual_action = run_quotation_denormalization_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman"Has God really said, '
            + "You shall not eat of any tree of the garden?"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationDenormalizationAction.SKIP
    assert actual_action == expected_action


def test_choose_best_action_based_on_observed_issues() -> None:
    first_pass_analyzer = QuotationDenormalizationFirstPass(QuoteConvention("", []), QuoteConvention("", []))
    first_pass_analyzer._will_basic_denormalization_work = False

    # Test with no issues
    best_action = first_pass_analyzer._choose_best_action_based_on_observed_issues([])
    assert best_action == QuotationDenormalizationAction.APPLY_FULL

    # Test with one issue
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.TOO_DEEP_NESTING]
        )
        == QuotationDenormalizationAction.SKIP
    )

    # Test with multiple issues
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
            ]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
            ]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationDenormalizationAction.SKIP
    )


def test_choose_best_action_based_on_observed_issues_with_basic_fallback() -> None:
    first_pass_analyzer = QuotationDenormalizationFirstPass(QuoteConvention("", []), QuoteConvention("", []))
    first_pass_analyzer._will_basic_denormalization_work = True

    # Test with no issues
    best_action = first_pass_analyzer._choose_best_action_based_on_observed_issues([])
    assert best_action == QuotationDenormalizationAction.APPLY_FULL

    # Test with one issue
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK]
        )
        == QuotationDenormalizationAction.APPLY_BASIC
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [QuotationMarkResolutionIssue.TOO_DEEP_NESTING]
        )
        == QuotationDenormalizationAction.APPLY_BASIC
    )

    # Test with multiple issues
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
            ]
        )
        == QuotationDenormalizationAction.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_action_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationDenormalizationAction.APPLY_BASIC
    )


# tests of get_best_actions_by_chapter()
def test_no_issues_in_usfm() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_FULL]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_opening_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_closing_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_too_deep_nesting() -> None:
    normalized_usfm = """\\c 1
    \\v 1 "Now the serpent was more "subtle than any animal
    of the "field which "Yahweh God had made.
    He said to the woman, "Has God really said,
    "You shall not eat of any tree of the garden?
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman"Has God really said,
    You shall not eat of any tree of the garden?
    """
    expected_actions = [QuotationDenormalizationAction.SKIP]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_no_issues_in_multiple_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, "Has God really said,
    'You shall not eat of any tree of the garden'?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_FULL]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_second_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_first_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had" made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    "You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_FULL]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_second_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not"eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_FULL, QuotationDenormalizationAction.SKIP]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_first_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field"which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    "You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.SKIP, QuotationDenormalizationAction.APPLY_FULL]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_both_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had" made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_both_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had"made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any"tree of the garden?
    """
    expected_actions = [QuotationDenormalizationAction.SKIP, QuotationDenormalizationAction.SKIP]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_in_first_ambiguous_in_second() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made."
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any"tree of the garden?
    """
    expected_actions = [QuotationDenormalizationAction.APPLY_BASIC, QuotationDenormalizationAction.SKIP]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_in_first_unpaired_in_second() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God"had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationDenormalizationAction.SKIP, QuotationDenormalizationAction.APPLY_BASIC]
    observed_actions = run_quotation_denormalization_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def run_quotation_denormalization_first_pass(
    normalized_usfm: str, source_quote_convention_name: str, target_quote_convention_name: str
) -> List[QuotationDenormalizationAction]:
    source_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    first_pass_analyzer = QuotationDenormalizationFirstPass(source_quote_convention, target_quote_convention)
    parse_usfm(normalized_usfm, first_pass_analyzer)

    return first_pass_analyzer.get_best_actions_by_chapter(normalized_usfm)


def run_quotation_denormalization_first_pass_on_chapter(
    verse_texts: List[str], source_quote_convention_name: str, target_quote_convention_name: str
) -> QuotationDenormalizationAction:
    source_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    first_pass_analyzer = QuotationDenormalizationFirstPass(source_quote_convention, target_quote_convention)

    chapter = Chapter([Verse([TextSegment.Builder().set_text(verse_text).build() for verse_text in verse_texts])])

    return first_pass_analyzer._find_best_action_for_chapter(chapter)


def get_quote_convention_by_name(name: str) -> QuoteConvention:
    quote_convention: Union[QuoteConvention, None] = (
        standard_quote_conventions.standard_quote_conventions.get_quote_convention_by_name(name)
    )
    assert quote_convention is not None
    return quote_convention
