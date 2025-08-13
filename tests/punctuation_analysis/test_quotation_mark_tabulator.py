# QuotationMarkCounts tests
from pytest import approx

from machine.punctuation_analysis import (
    QuotationMarkCounts,
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuotationMarkTabulator,
    QuoteConvention,
    SingleLevelQuoteConvention,
    TextSegment,
)


def test_get_observed_count() -> None:
    counts = QuotationMarkCounts()
    assert counts.get_observed_count() == 0

    counts.count_quotation_mark('"')
    assert counts.get_observed_count() == 1

    counts.count_quotation_mark('"')
    assert counts.get_observed_count() == 2

    counts.count_quotation_mark("'")
    assert counts.get_observed_count() == 3


def test_get_best_proportion() -> None:
    counts = QuotationMarkCounts()
    counts.count_quotation_mark('"')
    counts.count_quotation_mark('"')
    counts.count_quotation_mark("'")

    best_str, best_count, total_count = counts.find_best_quotation_mark_proportion()
    assert best_str == '"'
    assert best_count == 2
    assert total_count == 3

    counts.count_quotation_mark("'")
    counts.count_quotation_mark("'")

    best_str, best_count, total_count = counts.find_best_quotation_mark_proportion()
    assert best_str == "'"
    assert best_count == 3
    assert total_count == 5


def test_calculate_num_differences() -> None:
    counts = QuotationMarkCounts()
    counts.count_quotation_mark('"')
    counts.count_quotation_mark('"')
    counts.count_quotation_mark("'")

    assert counts.calculate_num_differences('"') == 1
    assert counts.calculate_num_differences("'") == 2
    assert counts.calculate_num_differences("\u201c") == 3

    counts.count_quotation_mark("'")
    assert counts.calculate_num_differences('"') == 2
    assert counts.calculate_num_differences("'") == 2
    assert counts.calculate_num_differences("\u201c") == 4


# QuotationMarkTabulator tests
def test_calculate_similarity() -> None:
    single_level_quotation_mark_tabulator = QuotationMarkTabulator()
    single_level_quotation_mark_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 0, 1),
        ]
    )

    assert (
        single_level_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention("", [SingleLevelQuoteConvention("\u201c", "\u201d")])
        )
        == 1.0
    )
    assert (
        single_level_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention("", [SingleLevelQuoteConvention("\u201d", "\u201c")])
        )
        == 0.0
    )
    assert (
        single_level_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention("", [SingleLevelQuoteConvention("\u201c", '"')])
        )
        == 0.5
    )
    assert (
        single_level_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention(
                "", [SingleLevelQuoteConvention("\u201c", "\u201d"), SingleLevelQuoteConvention("\u00ab", "\u00bb")]
            )
        )
        == 1.0
    )

    empty_quotation_mark_tabulator = QuotationMarkTabulator()
    assert (
        empty_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention("", [SingleLevelQuoteConvention("\u201c", "\u201d")])
        )
        == 0.0
    )

    two_level_quotation_mark_tabulator = QuotationMarkTabulator()
    two_level_quotation_mark_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 2),
            QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 0, 2),
        ]
    )
    assert two_level_quotation_mark_tabulator.calculate_similarity(
        QuoteConvention("", [SingleLevelQuoteConvention("\u201c", "\u201d")])
    ) == approx(0.66666666666667, rel=1e-9)
    assert (
        two_level_quotation_mark_tabulator.calculate_similarity(
            QuoteConvention(
                "", [SingleLevelQuoteConvention("\u201c", "\u201d"), SingleLevelQuoteConvention("\u2018", "\u2019")]
            )
        )
        == 1.0
    )
    assert two_level_quotation_mark_tabulator.calculate_similarity(
        QuoteConvention(
            "", [SingleLevelQuoteConvention("\u201c", "\u201d"), SingleLevelQuoteConvention("\u00ab", "\u00bb")]
        )
    ) == approx(0.66666666666667, rel=1e-9)
    assert two_level_quotation_mark_tabulator.calculate_similarity(
        QuoteConvention(
            "", [SingleLevelQuoteConvention("\u2018", "\u2019"), SingleLevelQuoteConvention("\u2018", "\u2019")]
        )
    ) == approx(0.33333333333333, rel=1e-9)
