from machine.corpora.analysis.quote_convention import QuoteConvention, SingleLevelQuoteConvention


def test_print_summary():
    quote_convention = QuoteConvention(
        "test-quote-convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201D"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201D", "\u201D"),
        ],
    )
    expected_summary_message = (
        "test-quote-convention\n"
        + "\u201CFirst-level quote\u201D\n"
        + "\u2018Second-level quote\u2019\n"
        + "\u201DThird-level quote\u201D\n"
    )
    assert quote_convention._get_summary_message() == expected_summary_message
    assert True
