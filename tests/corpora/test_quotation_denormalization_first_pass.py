from typing import List

from machine.corpora import QuotationDenormalizationAction, QuotationDenormalizationFirstPass, parse_usfm
from machine.corpora.analysis import standard_quote_conventions


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
