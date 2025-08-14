from machine.punctuation_analysis import QuotationMarkDirection
from machine.punctuation_analysis.quote_convention import QuoteConvention, SingleLevelQuoteConvention


def test_single_level_quote_convention_normalize() -> None:
    english_level1_quote_convention = SingleLevelQuoteConvention("\u201c", "\u201d")
    normalized_english_level1_quote_convention = english_level1_quote_convention.normalize()
    assert normalized_english_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_english_level1_quote_convention.closing_quotation_mark == '"'

    english_level2_quote_convention = SingleLevelQuoteConvention("\u2018", "\u2019")
    normalized_english_level2_quote_convention = english_level2_quote_convention.normalize()
    assert normalized_english_level2_quote_convention.opening_quotation_mark == "'"
    assert normalized_english_level2_quote_convention.closing_quotation_mark == "'"

    already_normalized_english_level1_quote_convention = SingleLevelQuoteConvention('"', '"')
    doubly_normalized_english_level1_quote_convention = already_normalized_english_level1_quote_convention.normalize()
    assert doubly_normalized_english_level1_quote_convention.opening_quotation_mark == '"'
    assert doubly_normalized_english_level1_quote_convention.closing_quotation_mark == '"'

    already_normalized_english_level2_quote_convention = SingleLevelQuoteConvention("'", "'")
    doubly_normalized_english_level2_quote_convention = already_normalized_english_level2_quote_convention.normalize()
    assert doubly_normalized_english_level2_quote_convention.opening_quotation_mark == "'"
    assert doubly_normalized_english_level2_quote_convention.closing_quotation_mark == "'"

    french_level1_quote_convention = SingleLevelQuoteConvention("\u00ab", "\u00bb")
    normalized_french_level1_quote_convention = french_level1_quote_convention.normalize()
    assert normalized_french_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_french_level1_quote_convention.closing_quotation_mark == '"'

    french_level2_quote_convention = SingleLevelQuoteConvention("\u2039", "\u203a")
    normalized_french_level2_quote_convention = french_level2_quote_convention.normalize()
    assert normalized_french_level2_quote_convention.opening_quotation_mark == "\u2039"
    assert normalized_french_level2_quote_convention.closing_quotation_mark == "\u203a"

    typewriter_french_level1_quote_convention = SingleLevelQuoteConvention("<<", ">>")
    normalized_typewriter_french_level1_quote_convention = typewriter_french_level1_quote_convention.normalize()
    assert normalized_typewriter_french_level1_quote_convention.opening_quotation_mark == "<<"
    assert normalized_typewriter_french_level1_quote_convention.closing_quotation_mark == ">>"

    typewriter_french_level2_quote_convention = SingleLevelQuoteConvention("<", ">")
    normalized_typewriter_french_level2_quote_convention = typewriter_french_level2_quote_convention.normalize()
    assert normalized_typewriter_french_level2_quote_convention.opening_quotation_mark == "<"
    assert normalized_typewriter_french_level2_quote_convention.closing_quotation_mark == ">"

    central_european_level1_quote_convention = SingleLevelQuoteConvention("\u201e", "\u201c")
    normalized_central_european_level1_quote_convention = central_european_level1_quote_convention.normalize()
    assert normalized_central_european_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_central_european_level1_quote_convention.closing_quotation_mark == '"'

    central_european_level2_quote_convention = SingleLevelQuoteConvention("\u201a", "\u2018")
    normalized_central_european_level2_quote_convention = central_european_level2_quote_convention.normalize()
    assert normalized_central_european_level2_quote_convention.opening_quotation_mark == "'"
    assert normalized_central_european_level2_quote_convention.closing_quotation_mark == "'"

    central_european_guillemets_quote_convention = SingleLevelQuoteConvention("\u00bb", "\u00ab")
    normalized_central_european_guillemets_quote_convention = central_european_guillemets_quote_convention.normalize()
    assert normalized_central_european_guillemets_quote_convention.opening_quotation_mark == '"'
    assert normalized_central_european_guillemets_quote_convention.closing_quotation_mark == '"'

    swedish_level1_quote_convention = SingleLevelQuoteConvention("\u201d", "\u201d")
    normalized_swedish_level1_quote_convention = swedish_level1_quote_convention.normalize()
    assert normalized_swedish_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_swedish_level1_quote_convention.closing_quotation_mark == '"'

    swedish_level2_quote_convention = SingleLevelQuoteConvention("\u2019", "\u2019")
    normalized_swedish_level2_quote_convention = swedish_level2_quote_convention.normalize()
    assert normalized_swedish_level2_quote_convention.opening_quotation_mark == "'"
    assert normalized_swedish_level2_quote_convention.closing_quotation_mark == "'"

    finnish_level1_quote_convention = SingleLevelQuoteConvention("\u00bb", "\u00bb")
    normalized_finnish_level1_quote_convention = finnish_level1_quote_convention.normalize()
    assert normalized_finnish_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_finnish_level1_quote_convention.closing_quotation_mark == '"'

    arabic_level1_quote_convention = SingleLevelQuoteConvention("\u201d", "\u201c")
    normalized_arabic_level1_quote_convention = arabic_level1_quote_convention.normalize()
    assert normalized_arabic_level1_quote_convention.opening_quotation_mark == '"'
    assert normalized_arabic_level1_quote_convention.closing_quotation_mark == '"'

    arabic_level2_quote_convention = SingleLevelQuoteConvention("\u2019", "\u2018")
    normalized_arabic_level2_quote_convention = arabic_level2_quote_convention.normalize()
    assert normalized_arabic_level2_quote_convention.opening_quotation_mark == "'"
    assert normalized_arabic_level2_quote_convention.closing_quotation_mark == "'"


def test_get_num_levels() -> None:
    empty_quote_convention = QuoteConvention("empty-quote-convention", [])
    assert empty_quote_convention.num_levels == 0

    one_level_quote_convention = QuoteConvention(
        "one_level_quote_convention",
        [SingleLevelQuoteConvention("\u201c", "\u201d")],
    )
    assert one_level_quote_convention.num_levels == 1

    two_level_quote_convention = QuoteConvention(
        "two_level_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    assert two_level_quote_convention.num_levels == 2

    three_level_quote_convention = QuoteConvention(
        "three_level_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201D", "\u201D"),
        ],
    )
    assert three_level_quote_convention.num_levels == 3


def test_get_opening_quote_at_level() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert quote_convention.get_opening_quotation_mark_at_depth(1) == "\u201c"
    assert quote_convention.get_opening_quotation_mark_at_depth(2) == "\u2018"
    assert quote_convention.get_opening_quotation_mark_at_depth(3) == "\u00ab"


def test_get_closing_quote_at_level() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert quote_convention.get_closing_quotation_mark_at_depth(1) == "\u201d"
    assert quote_convention.get_closing_quotation_mark_at_depth(2) == "\u2019"
    assert quote_convention.get_closing_quotation_mark_at_depth(3) == "\u00bb"


def test_get_expected_quotation_mark() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert quote_convention.get_expected_quotation_mark(1, QuotationMarkDirection.OPENING) == "\u201c"
    assert quote_convention.get_expected_quotation_mark(1, QuotationMarkDirection.CLOSING) == "\u201d"
    assert quote_convention.get_expected_quotation_mark(2, QuotationMarkDirection.OPENING) == "\u2018"
    assert quote_convention.get_expected_quotation_mark(2, QuotationMarkDirection.CLOSING) == "\u2019"
    assert quote_convention.get_expected_quotation_mark(3, QuotationMarkDirection.OPENING) == "\u00ab"
    assert quote_convention.get_expected_quotation_mark(3, QuotationMarkDirection.CLOSING) == "\u00bb"
    assert quote_convention.get_expected_quotation_mark(4, QuotationMarkDirection.OPENING) == ""
    assert quote_convention.get_expected_quotation_mark(4, QuotationMarkDirection.CLOSING) == ""
    assert quote_convention.get_expected_quotation_mark(0, QuotationMarkDirection.OPENING) == ""
    assert quote_convention.get_expected_quotation_mark(0, QuotationMarkDirection.CLOSING) == ""


def test_includes_opening_quotation_mark() -> None:
    empty_quote_convention = QuoteConvention("empty quote convention", [])
    assert not empty_quote_convention._includes_opening_quotation_mark("\u201c")

    positive_quote_convention1 = QuoteConvention(
        "positive_quote_convention_1", [SingleLevelQuoteConvention("\u201c", "\u201d")]
    )
    assert positive_quote_convention1._includes_opening_quotation_mark("\u201c")

    negative_quote_convention1 = QuoteConvention(
        "negative_quote_convention_1", [SingleLevelQuoteConvention("\u2018", "\u2019")]
    )
    assert not negative_quote_convention1._includes_opening_quotation_mark("\u201c")

    negative_quote_convention2 = QuoteConvention(
        "negative_quote_convention_2", [SingleLevelQuoteConvention("\u201d", "\u201c")]
    )
    assert not negative_quote_convention2._includes_opening_quotation_mark("\u201c")

    positive_quote_convention2 = QuoteConvention(
        "positive_quote_convention_2",
        [SingleLevelQuoteConvention("\u201c", "\u201d"), SingleLevelQuoteConvention("\u2018", "\u2019")],
    )
    assert positive_quote_convention2._includes_opening_quotation_mark("\u201c")

    positive_quote_convention3 = QuoteConvention(
        "positive_quote_convention_3",
        [SingleLevelQuoteConvention("\u2018", "\u2019"), SingleLevelQuoteConvention("\u201c", "\u201d")],
    )
    assert positive_quote_convention3._includes_opening_quotation_mark("\u201c")

    negative_quote_convention3 = QuoteConvention(
        "negative quote convention 3",
        [
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert not negative_quote_convention3._includes_opening_quotation_mark("\u201c")


def test_includes_closing_quotation_mark() -> None:
    empty_quote_convention = QuoteConvention("empty quote convention", [])
    assert not empty_quote_convention._includes_closing_quotation_mark("\u201d")

    positive_quote_convention1 = QuoteConvention(
        "positive_quote_convention_1", [SingleLevelQuoteConvention("\u201c", "\u201d")]
    )
    assert positive_quote_convention1._includes_closing_quotation_mark("\u201d")

    negative_quote_convention1 = QuoteConvention(
        "negative_quote_convention_1", [SingleLevelQuoteConvention("\u2018", "\u2019")]
    )
    assert not negative_quote_convention1._includes_closing_quotation_mark("\u201d")

    negative_quote_convention2 = QuoteConvention(
        "negative_quote_convention_2", [SingleLevelQuoteConvention("\u201d", "\u201c")]
    )
    assert not negative_quote_convention2._includes_closing_quotation_mark("\u201d")

    positive_quote_convention2 = QuoteConvention(
        "positive_quote_convention_2",
        [SingleLevelQuoteConvention("\u201c", "\u201d"), SingleLevelQuoteConvention("\u2018", "\u2019")],
    )
    assert positive_quote_convention2._includes_closing_quotation_mark("\u201d")

    positive_quote_convention3 = QuoteConvention(
        "positive_quote_convention_3",
        [SingleLevelQuoteConvention("\u2018", "\u2019"), SingleLevelQuoteConvention("\u201c", "\u201d")],
    )
    assert positive_quote_convention3._includes_closing_quotation_mark("\u201d")

    negative_quote_convention3 = QuoteConvention(
        "negative_quote_convention_3",
        [
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert not negative_quote_convention3._includes_closing_quotation_mark("\u201d")


def test_get_possible_depths() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    assert quote_convention.get_possible_depths("\u201c", QuotationMarkDirection.OPENING) == {1, 3}
    assert quote_convention.get_possible_depths("\u201c", QuotationMarkDirection.CLOSING) == set()
    assert quote_convention.get_possible_depths("\u2018", QuotationMarkDirection.OPENING) == {2, 4}
    assert quote_convention.get_possible_depths("\u2018", QuotationMarkDirection.CLOSING) == set()
    assert quote_convention.get_possible_depths("\u201d", QuotationMarkDirection.OPENING) == set()
    assert quote_convention.get_possible_depths("\u201d", QuotationMarkDirection.CLOSING) == {1, 3}
    assert quote_convention.get_possible_depths("\u2019", QuotationMarkDirection.OPENING) == set()
    assert quote_convention.get_possible_depths("\u2019", QuotationMarkDirection.CLOSING) == {2, 4}
    assert quote_convention.get_possible_depths("\u00ab", QuotationMarkDirection.OPENING) == set()
    assert quote_convention.get_possible_depths("\u00ab", QuotationMarkDirection.CLOSING) == set()


def test_is_compatible_with_observed_quotation_marks() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
        ],
    )
    assert quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u2018"], ["\u201d", "\u2019"])
    assert quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u00ab"], ["\u201d", "\u00bb"])
    assert quote_convention.is_compatible_with_observed_quotation_marks(["\u201c"], ["\u201d", "\u2019"])
    assert quote_convention.is_compatible_with_observed_quotation_marks(["\u201c"], ["\u201d"])
    assert quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u00ab"], ["\u201d", "\u2019"])

    assert not quote_convention.is_compatible_with_observed_quotation_marks(["\u201d", "\u2019"], ["\u201c"])

    assert not quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u201e"], ["\u201d"])

    assert not quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u2018"], ["\u201d", "\u201f"])

    # must have observed the first-level quotes
    assert not quote_convention.is_compatible_with_observed_quotation_marks(["\u2018"], ["\u201d"])
    assert not quote_convention.is_compatible_with_observed_quotation_marks(["\u201c", "\u2018"], ["\u00ab"])


def test_normalize() -> None:
    empty_quote_convention = QuoteConvention("empty-quote-convention", [])
    normalized_empty_quote_convention = empty_quote_convention.normalize()
    assert normalized_empty_quote_convention.name == "empty-quote-convention_normalized"
    assert normalized_empty_quote_convention.num_levels == 0

    standard_english_quote_convention = QuoteConvention(
        "standard_english_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    normalized_standard_english_quote_convention = standard_english_quote_convention.normalize()
    assert normalized_standard_english_quote_convention.name == "standard_english_quote_convention_normalized"
    assert normalized_standard_english_quote_convention.num_levels == 4
    assert normalized_standard_english_quote_convention.get_opening_quotation_mark_at_depth(1) == '"'
    assert normalized_standard_english_quote_convention.get_closing_quotation_mark_at_depth(1) == '"'
    assert normalized_standard_english_quote_convention.get_opening_quotation_mark_at_depth(2) == "'"
    assert normalized_standard_english_quote_convention.get_closing_quotation_mark_at_depth(2) == "'"
    assert normalized_standard_english_quote_convention.get_opening_quotation_mark_at_depth(3) == '"'
    assert normalized_standard_english_quote_convention.get_closing_quotation_mark_at_depth(3) == '"'
    assert normalized_standard_english_quote_convention.get_opening_quotation_mark_at_depth(4) == "'"
    assert normalized_standard_english_quote_convention.get_closing_quotation_mark_at_depth(4) == "'"

    western_european_quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    normalized_western_european_quote_convention = western_european_quote_convention.normalize()
    assert normalized_western_european_quote_convention.name == "test_quote_convention_normalized"
    assert normalized_western_european_quote_convention.num_levels == 3
    assert normalized_western_european_quote_convention.get_opening_quotation_mark_at_depth(1) == '"'
    assert normalized_western_european_quote_convention.get_closing_quotation_mark_at_depth(1) == '"'
    assert normalized_western_european_quote_convention.get_opening_quotation_mark_at_depth(2) == '"'
    assert normalized_western_european_quote_convention.get_closing_quotation_mark_at_depth(2) == '"'
    assert normalized_western_european_quote_convention.get_opening_quotation_mark_at_depth(3) == "'"
    assert normalized_western_european_quote_convention.get_closing_quotation_mark_at_depth(3) == "'"

    hybrid_british_typewriter_english_quote_convention = QuoteConvention(
        "hybrid_british_typewriter_english_quote_convention",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
        ],
    )

    normalized_hybrid_british_typewriter_english_quote_convention = (
        hybrid_british_typewriter_english_quote_convention.normalize()
    )
    assert (
        normalized_hybrid_british_typewriter_english_quote_convention.name
        == "hybrid_british_typewriter_english_quote_convention_normalized"
    )
    assert normalized_hybrid_british_typewriter_english_quote_convention.num_levels == 3
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_opening_quotation_mark_at_depth(1) == '"'
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_closing_quotation_mark_at_depth(1) == '"'
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_opening_quotation_mark_at_depth(2) == "'"
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_closing_quotation_mark_at_depth(2) == "'"
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_opening_quotation_mark_at_depth(3) == '"'
    assert normalized_hybrid_british_typewriter_english_quote_convention.get_closing_quotation_mark_at_depth(3) == '"'


def test_print_summary() -> None:
    quote_convention = QuoteConvention(
        "test_quote_convention",
        [
            SingleLevelQuoteConvention("\u201c", "\u201D"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201D", "\u201D"),
        ],
    )
    expected_summary_message = (
        "test_quote_convention\n"
        + "\u201CFirst-level quote\u201D\n"
        + "\u2018Second-level quote\u2019\n"
        + "\u201DThird-level quote\u201D\n"
    )
    assert str(quote_convention) == expected_summary_message
