from typing import List, Union

from machine.corpora import QuotationMarkUpdateFirstPass, QuotationMarkUpdateStrategy, parse_usfm
from machine.corpora.punctuation_analysis import (
    Chapter,
    QuotationMarkResolutionIssue,
    QuoteConvention,
    TextSegment,
    Verse,
    standard_quote_conventions,
)


def test_check_whether_fallback_mode_will_work() -> None:

    first_pass_analyzer = QuotationMarkUpdateFirstPass(QuoteConvention("", []), QuoteConvention("", []))

    # Cases where we expect fallback mode to work
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_english"),
            get_quote_convention_by_name("standard_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_french"),
            get_quote_convention_by_name("british_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_western_european"),
            get_quote_convention_by_name("standard_russian"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_western_european_variant"),
            get_quote_convention_by_name("standard_arabic"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("central_european"),
            get_quote_convention_by_name("british_typewriter_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_swedish"),
            get_quote_convention_by_name("typewriter_french"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_finnish"),
            get_quote_convention_by_name("british_inspired_western_european"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("eastern_european"),
            get_quote_convention_by_name("central_european"),
        )
        is True
    )

    # Cases where we expect fallback mode to fail
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_english"),
            get_quote_convention_by_name("western_european"),
        )
        is False
    )

    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_french"),
            get_quote_convention_by_name("western_european"),
        )
        is False
    )

    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_french"),
            get_quote_convention_by_name("french_variant"),
        )
        is False
    )

    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("central_european"),
            get_quote_convention_by_name("typewriter_western_european"),
        )
        is False
    )

    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("eastern_european"),
            get_quote_convention_by_name("standard_russian"),
        )
        is False
    )


def test_check_whether_fallback_mode_will_work_with_normalized_conventions() -> None:

    first_pass_analyzer = QuotationMarkUpdateFirstPass(QuoteConvention("", []), QuoteConvention("", []))

    # Cases where we expect fallback mode to work
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_english").normalize(),
            get_quote_convention_by_name("standard_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_french").normalize(),
            get_quote_convention_by_name("british_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_western_european").normalize(),
            get_quote_convention_by_name("standard_russian"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_western_european_variant").normalize(),
            get_quote_convention_by_name("standard_arabic"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("central_european").normalize(),
            get_quote_convention_by_name("british_typewriter_english"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_swedish").normalize(),
            get_quote_convention_by_name("typewriter_french"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_finnish").normalize(),
            get_quote_convention_by_name("british_inspired_western_european"),
        )
        is True
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("eastern_european").normalize(),
            get_quote_convention_by_name("central_european"),
        )
        is True
    )

    # Cases where we expect fallback mode to fail
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("western_european").normalize(),
            get_quote_convention_by_name("standard_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("french_variant").normalize(),
            get_quote_convention_by_name("hybrid_typewriter_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("british_inspired_western_european").normalize(),
            get_quote_convention_by_name("standard_russian"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("typewriter_english").normalize(),
            get_quote_convention_by_name("western_european"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("central_european_guillemets").normalize(),
            get_quote_convention_by_name("french_variant"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_arabic").normalize(),
            get_quote_convention_by_name("hybrid_typewriter_english"),
        )
        is False
    )
    assert (
        first_pass_analyzer._check_whether_fallback_mode_will_work(
            get_quote_convention_by_name("standard_russian").normalize(),
            get_quote_convention_by_name("standard_french"),
        )
        is False
    )


def test_choose_best_action_for_chapter() -> None:
    # Verse text with no issues
    actual_action = run_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + "He said to the woman, “Has God really said, "
            + "‘You shall not eat of any tree of the garden’?”"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.APPLY_FULL
    assert actual_action == expected_action

    # Verse text with unpaired opening quotation mark
    actual_action = run_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + "He said to the woman, “Has God really said, "
            + "‘You shall not eat of any tree of the garden’?"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.APPLY_FALLBACK
    assert actual_action == expected_action

    # Verse text with unpaired closing quotation mark
    actual_action = run_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + "He said to the woman, Has God really said, "
            + "You shall not eat of any tree of the garden?”"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.APPLY_FALLBACK
    assert actual_action == expected_action

    # Verse text with too deeply nested quotation marks
    actual_action = run_first_pass_on_chapter(
        [
            "“Now the serpent was more “subtle than any animal "
            + "of the “field which “Yahweh God had made. "
            + "He said to the woman, “Has God really said, "
            + "“You shall not eat of any tree of the garden?"
        ],
        "standard_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.APPLY_FALLBACK
    assert actual_action == expected_action

    # Verse text with an ambiguous quotation mark
    actual_action = run_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman"Has God really said, '
            + "You shall not eat of any tree of the garden?"
        ],
        "typewriter_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.SKIP
    assert actual_action == expected_action

    # Verse text with an ambiguous quotation mark
    actual_action = run_first_pass_on_chapter(
        [
            "Now the serpent was more subtle than any animal "
            + "of the field which Yahweh God had made. "
            + 'He said to the woman"Has God really said, '
            + "You shall not eat of any tree of the garden?"
        ],
        "typewriter_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.SKIP
    assert actual_action == expected_action

    # Verse text with too deeply nested ambiguous quotation marks
    actual_action = run_first_pass_on_chapter(
        [
            '"Now the serpent was more "subtle than any animal '
            + 'of the "field which "Yahweh God had made. '
            + 'He said to the woman, "Has God really said, '
            + '"You shall not eat of any tree of the garden?'
        ],
        "typewriter_english",
        "standard_english",
    )
    expected_action = QuotationMarkUpdateStrategy.SKIP
    assert actual_action == expected_action


def test_choose_best_action_based_on_observed_issues() -> None:
    first_pass_analyzer = QuotationMarkUpdateFirstPass(QuoteConvention("", []), QuoteConvention("", []))
    first_pass_analyzer._will_fallback_mode_work = False

    # Test with no issues
    best_action = first_pass_analyzer._choose_best_strategy_based_on_observed_issues([])
    assert best_action == QuotationMarkUpdateStrategy.APPLY_FULL

    # Test with one issue
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.TOO_DEEP_NESTING]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )

    # Test with multiple issues
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
            ]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
            ]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )


def test_choose_best_action_based_on_observed_issues_with_basic_fallback() -> None:
    first_pass_analyzer = QuotationMarkUpdateFirstPass(QuoteConvention("", []), QuoteConvention("", []))
    first_pass_analyzer._will_fallback_mode_work = True

    # Test with no issues
    best_action = first_pass_analyzer._choose_best_strategy_based_on_observed_issues([])
    assert best_action == QuotationMarkUpdateStrategy.APPLY_FULL

    # Test with one issue
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK]
        )
        == QuotationMarkUpdateStrategy.APPLY_FALLBACK
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [QuotationMarkResolutionIssue.TOO_DEEP_NESTING]
        )
        == QuotationMarkUpdateStrategy.APPLY_FALLBACK
    )

    # Test with multiple issues
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK,
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
            ]
        )
        == QuotationMarkUpdateStrategy.SKIP
    )
    assert (
        first_pass_analyzer._choose_best_strategy_based_on_observed_issues(
            [
                QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
                QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
            ]
        )
        == QuotationMarkUpdateStrategy.APPLY_FALLBACK
    )


# tests of get_best_actions_by_chapter()
def test_no_issues_in_usfm() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FULL]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_opening_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_closing_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_too_deep_nesting() -> None:
    normalized_usfm = """\\c 1
    \\v 1 “Now the serpent was more “subtle than any animal
    of the “field which “Yahweh God had made.
    He said to the woman, “Has God really said,
    “You shall not eat of any tree of the garden?
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman"Has God really said,
    You shall not eat of any tree of the garden?
    """
    expected_actions = [QuotationMarkUpdateStrategy.SKIP]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def test_no_issues_in_multiple_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FULL, QuotationMarkUpdateStrategy.APPLY_FULL]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_second_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FULL, QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_first_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had” made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    “You shall not eat of any tree of the garden?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.APPLY_FULL]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_second_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not"eat of any tree of the garden?"
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FULL, QuotationMarkUpdateStrategy.SKIP]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_first_chapter() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field"which Yahweh God had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    "You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationMarkUpdateStrategy.SKIP, QuotationMarkUpdateStrategy.APPLY_FULL]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_quotation_mark_in_both_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had” made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?”
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "standard_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_quotation_mark_in_both_chapters() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had"made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any"tree of the garden?
    """
    expected_actions = [QuotationMarkUpdateStrategy.SKIP, QuotationMarkUpdateStrategy.SKIP]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def test_unpaired_in_first_ambiguous_in_second() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made."
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any"tree of the garden?
    """
    expected_actions = [QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.SKIP]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def test_ambiguous_in_first_unpaired_in_second() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God"had made.
    \\c 2 \\v 1 He said to the woman, Has God really said,
    You shall not eat of any tree of the garden?"
    """
    expected_actions = [QuotationMarkUpdateStrategy.SKIP, QuotationMarkUpdateStrategy.APPLY_FALLBACK]
    observed_actions = run_first_pass(normalized_usfm, "typewriter_english", "standard_english")

    assert expected_actions == observed_actions


def run_first_pass(
    normalized_usfm: str, source_quote_convention_name: str, target_quote_convention_name: str
) -> List[QuotationMarkUpdateStrategy]:
    source_quote_convention = standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    first_pass_analyzer = QuotationMarkUpdateFirstPass(source_quote_convention, target_quote_convention)
    parse_usfm(normalized_usfm, first_pass_analyzer)

    return first_pass_analyzer.find_best_chapter_strategies()


def run_first_pass_on_chapter(
    verse_texts: List[str], source_quote_convention_name: str, target_quote_convention_name: str
) -> QuotationMarkUpdateStrategy:
    source_quote_convention = standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
        source_quote_convention_name
    )
    assert source_quote_convention is not None

    target_quote_convention = standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
        target_quote_convention_name
    )
    assert target_quote_convention is not None

    first_pass_analyzer = QuotationMarkUpdateFirstPass(source_quote_convention, target_quote_convention)

    chapter = Chapter([Verse([TextSegment.Builder().set_text(verse_text).build() for verse_text in verse_texts])])

    return first_pass_analyzer._find_best_strategy_for_chapter(chapter)


def get_quote_convention_by_name(name: str) -> QuoteConvention:
    quote_convention: Union[QuoteConvention, None] = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(name)
    )
    assert quote_convention is not None
    return quote_convention
