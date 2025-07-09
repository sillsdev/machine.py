from machine.corpora.punctuation_analysis import (
    ApostropheProportionStatistics,
    Chapter,
    PreliminaryApostropheAnalyzer,
    PreliminaryQuotationAnalyzer,
    QuotationMarkGrouper,
    QuotationMarkSequences,
    QuotationMarkStringMatch,
    QuotationMarkWordPositions,
    QuoteConvention,
    QuoteConventionSet,
    SingleLevelQuoteConvention,
    TextSegment,
    Verse,
)


# ApostropheProportionStatistics tests
def test_apostrophe_proportion_statistics_reset() -> None:
    apostrophe_proportion_statistics = ApostropheProportionStatistics()
    apostrophe_proportion_statistics.count_characters(TextSegment.Builder().set_text("'").build())
    apostrophe_proportion_statistics.add_apostrophe()
    assert apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.5)

    apostrophe_proportion_statistics.reset()
    assert not apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.5)


def test_is_apostrophe_proportion_greater_than() -> None:
    apostrophe_proportion_statistics = ApostropheProportionStatistics()
    assert not apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.0)

    # invalid case where no characters have been counted
    apostrophe_proportion_statistics.add_apostrophe()
    assert not apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.0)

    apostrophe_proportion_statistics.count_characters(TextSegment.Builder().set_text("a").build())
    assert apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.99)

    apostrophe_proportion_statistics.add_apostrophe()
    apostrophe_proportion_statistics.count_characters(TextSegment.Builder().set_text("bcd").build())
    assert apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.4)
    assert not apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.5)

    apostrophe_proportion_statistics.count_characters(TextSegment.Builder().set_text("ef").build())
    assert apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.3)
    assert not apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(0.4)


# QuotationMarkWordPosition tests
def test_is_mark_rarely_initial() -> None:
    quotation_mark_word_positions = QuotationMarkWordPositions()
    assert not quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    assert quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    assert not quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    assert quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201c")
    assert quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201c")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201c")
    assert quotation_mark_word_positions.is_mark_rarely_initial("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    assert not quotation_mark_word_positions.is_mark_rarely_initial("\u201d")


def test_is_mark_rarely_final() -> None:
    quotation_mark_word_positions = QuotationMarkWordPositions()
    assert not quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    assert quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    assert not quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    assert quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201c")
    assert quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201c")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201c")
    assert quotation_mark_word_positions.is_mark_rarely_final("\u201d")

    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    assert not quotation_mark_word_positions.is_mark_rarely_final("\u201d")


def test_are_initial_and_final_rates_similar() -> None:
    quotation_mark_word_positions = QuotationMarkWordPositions()
    assert not quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    assert quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    assert not quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    assert quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")

    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    assert not quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")

    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    assert quotation_mark_word_positions.are_initial_and_final_rates_similar("\u201d")


def test_is_mark_commonly_mid_word() -> None:
    quotation_mark_word_positions = QuotationMarkWordPositions()
    assert not quotation_mark_word_positions.is_mark_commonly_mid_word("'")

    quotation_mark_word_positions.count_mid_word_apostrophe("'")
    assert quotation_mark_word_positions.is_mark_commonly_mid_word("'")

    quotation_mark_word_positions.count_word_initial_apostrophe("'")
    quotation_mark_word_positions.count_word_final_apostrophe("'")
    quotation_mark_word_positions.count_word_initial_apostrophe("'")
    quotation_mark_word_positions.count_word_final_apostrophe("'")
    assert not quotation_mark_word_positions.is_mark_commonly_mid_word("'")

    quotation_mark_word_positions.count_mid_word_apostrophe("'")
    assert quotation_mark_word_positions.is_mark_commonly_mid_word("'")


def test_quotation_mark_word_positions_reset() -> None:
    quotation_mark_word_positions = QuotationMarkWordPositions()
    quotation_mark_word_positions.count_word_initial_apostrophe("\u201d")
    quotation_mark_word_positions.count_word_final_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")
    quotation_mark_word_positions.count_mid_word_apostrophe("\u201d")

    assert quotation_mark_word_positions.is_mark_commonly_mid_word("\u201d")

    quotation_mark_word_positions.reset()

    assert not quotation_mark_word_positions.is_mark_commonly_mid_word("\u201d")


# QuotationMarkSequence tests
def test_is_mark_much_more_common_earlier() -> None:
    quotation_mark_sequences = QuotationMarkSequences()
    assert not quotation_mark_sequences.is_mark_much_more_common_earlier('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_much_more_common_earlier('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    assert not quotation_mark_sequences.is_mark_much_more_common_earlier('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_much_more_common_earlier('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    assert not quotation_mark_sequences.is_mark_much_more_common_earlier('"')


def test_is_mark_much_more_common_later() -> None:
    quotation_mark_sequences = QuotationMarkSequences()
    assert not quotation_mark_sequences.is_mark_much_more_common_later('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_much_more_common_later('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    assert not quotation_mark_sequences.is_mark_much_more_common_later('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_much_more_common_later('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    assert not quotation_mark_sequences.is_mark_much_more_common_later('"')


def test_is_mark_common_early_and_late() -> None:
    quotation_mark_sequences = QuotationMarkSequences()
    assert not quotation_mark_sequences.is_mark_common_early_and_late('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_common_early_and_late('"')

    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    quotation_mark_sequences.record_earlier_quotation_mark('"')
    quotation_mark_sequences.record_later_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_common_early_and_late('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    assert quotation_mark_sequences.is_mark_common_early_and_late('"')

    quotation_mark_sequences.record_later_quotation_mark('"')
    assert not quotation_mark_sequences.is_mark_common_early_and_late('"')


# QuotationMarkGrouper tests
def test_get_quotation_mark_pairs() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    typewriter_english_quote_convention: QuoteConvention = QuoteConvention(
        "typewriter_english",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )

    quotation_mark_grouper = QuotationMarkGrouper([], QuoteConventionSet([standard_english_quote_convention]))
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # no paired quotation mark
    quotation_mark_grouper = QuotationMarkGrouper(
        [QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # basic quotation mark pair
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == [("\u201c", "\u201d")]

    # out-of-order quotation mark pair
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # multiple unpaired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2019").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2019").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # paired and unpaired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2018\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2018\u201d").build(), 1, 2),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2018\u201d").build(), 2, 3),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == [("\u201c", "\u201d")]

    # ambiguous unpaired quotation mark
    quotation_mark_grouper = QuotationMarkGrouper(
        [QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)],
        QuoteConventionSet([typewriter_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # paired ambiguous quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text('""').build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text('""').build(), 1, 2),
        ],
        QuoteConventionSet([typewriter_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == [('"', '"')]

    # multiple paired quotation marks (should be skipped because we don't know how to pair them)
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u201c\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u201c\u201d").build(), 1, 2),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u201c\u201d").build(), 2, 3),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u201c\u201d").build(), 3, 4),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []

    # multiple different paired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u2018\u2019").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u2018\u2019").build(), 1, 2),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u2018\u2019").build(), 2, 3),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d\u2018\u2019").build(), 3, 4),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == [("\u201c", "\u201d"), ("\u2018", "\u2019")]

    # second-level paired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018\u2019").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018\u2019").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == [("\u2018", "\u2019")]

    # quotation marks that don't match the convention set
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2),
        ],
        QuoteConventionSet([typewriter_english_quote_convention]),
    )
    assert list(quotation_mark_grouper.get_quotation_mark_pairs()) == []


def test_has_distinct_paired_quotation_marks() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    typewriter_english_quote_convention: QuoteConvention = QuoteConvention(
        "typewriter_english",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )

    quotation_mark_grouper = QuotationMarkGrouper(
        [],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201c")
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201d")
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("")

    # basic paired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201c")
    assert quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201d")

    # second-level paired quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018\u2019").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018\u2019").build(), 1, 2),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert quotation_mark_grouper.has_distinct_paired_quotation_mark("\u2018")
    assert quotation_mark_grouper.has_distinct_paired_quotation_mark("\u2019")

    # only one half of the pair observed
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        ],
        QuoteConventionSet([standard_english_quote_convention]),
    )
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201c")
    assert quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201d")

    # quotation marks that don't match the convention set
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2),
        ],
        QuoteConventionSet([typewriter_english_quote_convention]),
    )
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201c")
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark("\u201d")

    # ambiguous quotation marks
    quotation_mark_grouper = QuotationMarkGrouper(
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text('""').build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text('""').build(), 1, 2),
        ],
        QuoteConventionSet([typewriter_english_quote_convention]),
    )
    assert not quotation_mark_grouper.has_distinct_paired_quotation_mark('"')


# PreliminaryApostropheAnalyzer tests
def test_that_the_mark_must_be_an_apostrophe() -> None:
    preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(
                TextSegment.Builder().set_text("alternative mid\u2019word apostrophe").build(), 15, 16
            ),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid\u2018word quotation mark").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid\u201cword quotation mark").build(), 3, 4),
        ],
    )
    assert preliminary_apostrophe_analyzer.is_apostrophe_only("'")
    assert preliminary_apostrophe_analyzer.is_apostrophe_only("\u2019")
    assert not preliminary_apostrophe_analyzer.is_apostrophe_only("\u2018")
    assert not preliminary_apostrophe_analyzer.is_apostrophe_only("\u201c")
    assert not preliminary_apostrophe_analyzer.is_apostrophe_only("\u201d")


def test_that_a_rarely_initial_or_final_mark_is_an_apostrophe() -> None:
    negative_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    negative_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
        ],
    )
    assert not negative_preliminary_apostrophe_analyzer.is_apostrophe_only("'")

    positive_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    positive_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
            TextSegment.Builder()
            .set_text(
                "The proportion must be kept below 0.02, because quotation marks should occur relatively infrequently"
            )
            .build(),
            TextSegment.Builder()
            .set_text(
                "Apostrophes, on the other hand, can be much more common, especially in non-English languages where they "
                + "can indicate a glottal stop"
            )
            .build(),
            TextSegment.Builder()
            .set_text("Technically Unicode has a separate character for the glottal stop, but it is rarely used")
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
        ],
    )
    assert positive_preliminary_apostrophe_analyzer.is_apostrophe_only("'")


def test_that_a_mark_with_similar_final_and_initial_rates_is_an_apostrophe() -> None:
    negative_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    negative_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
            TextSegment.Builder()
            .set_text("We need a ton of text here to keep the proportion low, since we have 8 apostrophes in this test")
            .build(),
            TextSegment.Builder()
            .set_text(
                "The proportion must be kept below 0.02, because quotation marks should occur relatively infrequently"
            )
            .build(),
            TextSegment.Builder()
            .set_text(
                "Apostrophes, on the other hand, can be much more common, especially in non-English languages where they "
                + "can indicate a glottal stop"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
        ],
    )
    assert not negative_preliminary_apostrophe_analyzer.is_apostrophe_only("'")

    negative_preliminary_apostrophe_analyzer2 = PreliminaryApostropheAnalyzer()
    negative_preliminary_apostrophe_analyzer2.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
            TextSegment.Builder()
            .set_text("We need a ton of text here to keep the proportion low, since we have 8 apostrophes in this test")
            .build(),
            TextSegment.Builder()
            .set_text(
                "The proportion must be kept below 0.02, because quotation marks should occur relatively infrequently"
            )
            .build(),
            TextSegment.Builder()
            .set_text(
                "Apostrophes, on the other hand, can be much more common, especially in non-English languages where they "
                + "can indicate a glottal stop"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
        ],
    )
    assert not negative_preliminary_apostrophe_analyzer2.is_apostrophe_only("'")

    positive_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    positive_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
            TextSegment.Builder()
            .set_text("We need a ton of text here to keep the proportion low, since we have 8 apostrophes in this test")
            .build(),
            TextSegment.Builder()
            .set_text(
                "The proportion must be kept below 0.02, because quotation marks should occur relatively infrequently"
            )
            .build(),
            TextSegment.Builder()
            .set_text(
                "Apostrophes, on the other hand, can be much more common, especially in non-English languages where they "
                + "can indicate a glottal stop"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
        ],
    )
    assert positive_preliminary_apostrophe_analyzer.is_apostrophe_only("'")


def test_that_a_commonly_mid_word_mark_is_an_apostrophe() -> None:
    negative_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    negative_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
        ],
    )
    assert not negative_preliminary_apostrophe_analyzer.is_apostrophe_only("'")

    positive_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    positive_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("mid'word apostrophe").build(), 3, 4),
        ],
    )
    assert positive_preliminary_apostrophe_analyzer.is_apostrophe_only("'")


def test_that_a_frequently_occurring_character_is_an_apostrophe() -> None:
    negative_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    negative_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Long text segment to help keep the proportion of apostrophes low").build(),
            TextSegment.Builder()
            .set_text(
                "If a mark appears very frequently in the text, it is likely an apostrophe, instead of a quotation mark"
            )
            .build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
        ],
    )
    assert not negative_preliminary_apostrophe_analyzer.is_apostrophe_only("'")

    positive_preliminary_apostrophe_analyzer = PreliminaryApostropheAnalyzer()
    positive_preliminary_apostrophe_analyzer.process_quotation_marks(
        [
            TextSegment.Builder().set_text("Very short text").build(),
        ],
        [
            QuotationMarkStringMatch(TextSegment.Builder().set_text("'word initial apostrophe").build(), 0, 1),
            QuotationMarkStringMatch(TextSegment.Builder().set_text("word' final apostrophe").build(), 4, 5),
        ],
    )
    assert positive_preliminary_apostrophe_analyzer.is_apostrophe_only("'")


# PreliminaryQuotationAnalyzer tests
def test_that_quotation_mark_sequence_is_used_to_determine_opening_and_closing_quotes() -> None:
    standard_english_quote_convention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    typewriter_english_quote_convention = QuoteConvention(
        "typewriter_english",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )
    standard_french_quote_convention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    western_european_quote_convention = QuoteConvention(
        "western_european",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    standard_swedish_quote_convention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    preliminary_quotation_analyzer = PreliminaryQuotationAnalyzer(
        QuoteConventionSet(
            [
                standard_english_quote_convention,
                typewriter_english_quote_convention,
                standard_french_quote_convention,
                western_european_quote_convention,
                standard_swedish_quote_convention,
            ]
        )
    )

    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("initial text \u201c quoted English text \u201d final text")
                            .build()
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([standard_english_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("initial text \u201d quoted Swedish text \u201d final text")
                            .build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([standard_swedish_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("initial text \u00ab quoted French/Western European text \u00bb final text")
                            .build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([standard_french_quote_convention, western_european_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text('initial text " quoted typewriter English text " final text')
                            .build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([typewriter_english_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("initial text \u201c quoted English text \u201d final text")
                            .build(),
                            TextSegment.Builder().set_text("second level \u2018 English quotes \u2019").build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([standard_english_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text('initial text " quoted typewriter English text " final text')
                            .build(),
                            TextSegment.Builder().set_text("second level 'typewriter quotes'").build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([typewriter_english_quote_convention])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("initial text \u201c quoted English text \u201d final text")
                            .build(),
                            TextSegment.Builder()
                            .set_text("the quotes \u201d in this segment \u201c are backwards")
                            .build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([])

    preliminary_quotation_analyzer.reset()
    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("first-level quotes \u2018 must be observed \u2019 to retain a quote convention")
                            .build(),
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([])


def test_that_apostrophes_not_considered_as_quotation_marks() -> None:
    standard_english_quote_convention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    typewriter_english_quote_convention = QuoteConvention(
        "typewriter_english",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )

    preliminary_quotation_analyzer = PreliminaryQuotationAnalyzer(
        QuoteConventionSet(
            [
                standard_english_quote_convention,
                typewriter_english_quote_convention,
            ]
        )
    )

    assert preliminary_quotation_analyzer.narrow_down_possible_quote_conventions(
        [
            Chapter(
                [
                    Verse(
                        [
                            TextSegment.Builder()
                            .set_text("ini'tial 'text \u201c quo'ted English text' \u201d fi'nal text")
                            .build()
                        ]
                    )
                ]
            )
        ]
    ) == QuoteConventionSet([standard_english_quote_convention])
