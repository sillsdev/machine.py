import regex

from machine.punctuation_analysis import (
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuotationMarkStringMatch,
    QuoteConvention,
    QuoteConventionSet,
    SingleLevelQuoteConvention,
    TextSegment,
    UsfmMarkerType,
)


def test_get_quotation_mark() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("quick brown fox").build(), 6, 7
    )
    assert quotation_mark_string_match.quotation_mark == "b"

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("quick brown fox").build(), 6, 10
    )
    assert quotation_mark_string_match.quotation_mark == "brow"

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("q").build(), 0, 1)
    assert quotation_mark_string_match.quotation_mark == "q"


def test_is_valid_opening_quotation_mark() -> None:
    standard_english_quote_convention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    assert quotation_mark_string_match.is_valid_opening_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    assert not quotation_mark_string_match.is_valid_opening_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 1, 2)
    assert quotation_mark_string_match.is_valid_opening_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 0, 2)
    assert not quotation_mark_string_match.is_valid_opening_quotation_mark(standard_english_quote_convention_set)


def test_is_valid_closing_quotation_mark() -> None:
    standard_english_quote_convention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )
    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    assert quotation_mark_string_match.is_valid_closing_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    assert not quotation_mark_string_match.is_valid_closing_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 0, 1)
    assert quotation_mark_string_match.is_valid_closing_quotation_mark(standard_english_quote_convention_set)

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d\u201c").build(), 0, 2)
    assert not quotation_mark_string_match.is_valid_closing_quotation_mark(standard_english_quote_convention_set)


def test_does_quotation_mark_match() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert quotation_mark_string_match.quotation_mark_matches(regex.compile(r"^s$"))
    assert not quotation_mark_string_match.quotation_mark_matches(regex.compile(r"a"))
    assert not quotation_mark_string_match.quotation_mark_matches(regex.compile(r"sa"))


def test_does_next_character_match() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.next_character_matches(regex.compile(r"^s$"))
    assert quotation_mark_string_match.next_character_matches(regex.compile(r"a"))
    assert not quotation_mark_string_match.next_character_matches(regex.compile(r"sa"))

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert not quotation_mark_string_match.next_character_matches(regex.compile(r".*"))


def test_does_previous_character_match() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert quotation_mark_string_match.previous_character_matches(regex.compile(r"^s$"))
    assert not quotation_mark_string_match.previous_character_matches(regex.compile(r"a"))
    assert not quotation_mark_string_match.previous_character_matches(regex.compile(r"sa"))

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.previous_character_matches(regex.compile(r".*"))


def test_get_previous_character() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert quotation_mark_string_match.previous_character == "s"

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert quotation_mark_string_match.previous_character == "x"

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert quotation_mark_string_match.previous_character is None

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2)
    assert quotation_mark_string_match.previous_character == "“"


def test_get_next_character() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert quotation_mark_string_match.next_character == "m"

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert quotation_mark_string_match.next_character == "a"

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert quotation_mark_string_match.next_character is None

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1)
    assert quotation_mark_string_match.next_character == "”"


def test_does_leading_substring_match() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 5, 6)
    assert quotation_mark_string_match.leading_substring_matches(regex.compile(r"^sampl$"))

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.leading_substring_matches(regex.compile(r".+"))

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2)
    assert quotation_mark_string_match.leading_substring_matches(regex.compile(r"\u201c"))


def test_does_trailing_substring_match() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 5, 6)
    assert quotation_mark_string_match.trailing_substring_matches(regex.compile(r"^ text$"))

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert not quotation_mark_string_match.trailing_substring_matches(regex.compile(r".+"))

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 0, 1)
    assert quotation_mark_string_match.trailing_substring_matches(regex.compile(r"\u201d"))


def test_get_context() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("this is a bunch' of sample text").build(), 15, 16
    )
    assert quotation_mark_string_match.context == "is a bunch' of sample"

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("this is a bunch' of sample text").build(), 5, 6
    )
    assert quotation_mark_string_match.context == "this is a bunch'"

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("this is a bunch' of sample text").build(), 25, 26
    )
    assert quotation_mark_string_match.context == "' of sample text"

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("short").build(), 3, 4)
    assert quotation_mark_string_match.context == "short"


def test_resolve() -> None:
    text_segment = TextSegment.Builder().set_text("'").build()
    quotation_mark_string_match = QuotationMarkStringMatch(text_segment, 0, 1)
    assert quotation_mark_string_match.resolve(2, QuotationMarkDirection.OPENING) == QuotationMarkMetadata(
        "'", 2, QuotationMarkDirection.OPENING, text_segment, 0, 1
    )
    assert quotation_mark_string_match.resolve(1, QuotationMarkDirection.OPENING) == QuotationMarkMetadata(
        "'", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1
    )
    assert quotation_mark_string_match.resolve(1, QuotationMarkDirection.CLOSING) == QuotationMarkMetadata(
        "'", 1, QuotationMarkDirection.CLOSING, text_segment, 0, 1
    )


def test_is_at_start_of_segment() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert quotation_mark_string_match.is_at_start_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert not quotation_mark_string_match.is_at_start_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("\u201csample text").build(), 0, 1
    )
    assert quotation_mark_string_match.is_at_start_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 15, 16
    )
    assert not quotation_mark_string_match.is_at_start_of_segment()


def test_is_at_end_of_segment() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert quotation_mark_string_match.is_at_end_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.is_at_end_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("\u201csample text\u201d").build(), 12, 13
    )
    assert quotation_mark_string_match.is_at_end_of_segment()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 15, 16
    )
    assert not quotation_mark_string_match.is_at_end_of_segment()


def test_has_leading_whitespace() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 7, 8)
    assert quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample\ttext").build(), 7, 8)
    assert quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
        0,
        1,
    )
    assert quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.EMBED).build(), 0, 1
    )
    assert quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.VERSE).build(), 0, 1
    )
    assert quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.CHAPTER).build(), 0, 1
    )
    assert not quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.CHARACTER).build(), 0, 1
    )
    assert not quotation_mark_string_match.has_leading_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("\u201csample text").add_preceding_marker(UsfmMarkerType.VERSE).build(),
        0,
        1,
    )
    assert quotation_mark_string_match.has_leading_whitespace()


def test_has_trailing_whitespace() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 5, 6)
    assert quotation_mark_string_match.has_trailing_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample\ttext").build(), 5, 6)
    assert quotation_mark_string_match.has_trailing_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert not quotation_mark_string_match.has_trailing_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
        10,
        11,
    )
    assert not quotation_mark_string_match.has_trailing_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.EMBED).build(), 10, 11
    )
    assert not quotation_mark_string_match.has_trailing_whitespace()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").add_preceding_marker(UsfmMarkerType.VERSE).build(), 10, 11
    )
    assert not quotation_mark_string_match.has_trailing_whitespace()


def test_has_leading_punctuation() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample)\u201d text").build(), 7, 8
    )
    assert quotation_mark_string_match.has_leading_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample) \u201d text").build(), 8, 9
    )
    assert not quotation_mark_string_match.has_leading_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample,\u201d text").build(), 7, 8
    )
    assert quotation_mark_string_match.has_leading_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample.\u201d text").build(), 7, 8
    )
    assert quotation_mark_string_match.has_leading_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("\u201csample text").build(), 0, 1
    )
    assert not quotation_mark_string_match.has_leading_punctuation()


def test_has_trailing_punctuation() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample \u201c-text").build(), 7, 8
    )
    assert quotation_mark_string_match.has_trailing_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample \u201c text").build(), 7, 8
    )
    assert not quotation_mark_string_match.has_trailing_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text\u201d").build(), 11, 12
    )
    assert not quotation_mark_string_match.has_trailing_punctuation()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample', text\u201d").build(), 6, 7
    )
    assert quotation_mark_string_match.has_trailing_punctuation()


def test_has_letter_in_leading_substring() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert quotation_mark_string_match.has_letter_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("ꮪample text").build(), 1, 2)
    assert quotation_mark_string_match.has_letter_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.has_letter_in_leading_substring()


def test_has_letter_in_trailing_substring() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 9, 10)
    assert quotation_mark_string_match.has_letter_in_trailing_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample tex𑢼").build(), 9, 10)
    assert quotation_mark_string_match.has_letter_in_trailing_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert not quotation_mark_string_match.has_letter_in_trailing_substring()


def test_has_leading_latin_letter() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 1, 2)
    assert quotation_mark_string_match.has_leading_latin_letter()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("5ample text").build(), 1, 2)
    assert not quotation_mark_string_match.has_leading_latin_letter()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("Ｓample text").build(), 1, 2)
    assert quotation_mark_string_match.has_leading_latin_letter()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 0, 1)
    assert not quotation_mark_string_match.has_leading_latin_letter()


def test_has_trailing_latin_letter() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample text").build(), 9, 10)
    assert quotation_mark_string_match.has_trailing_latin_letter()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample texＴ").build(), 9, 10
    )
    assert quotation_mark_string_match.has_trailing_latin_letter()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample text").build(), 10, 11
    )
    assert not quotation_mark_string_match.has_trailing_latin_letter()


def test_has_quote_introducer_in_leading_substring() -> None:
    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample, \u201ctext").build(), 8, 9
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample,\u201ctext").build(), 7, 8
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample: \u201ctext").build(), 8, 9
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample:\u201ctext").build(), 7, 8
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample,  \u201ctext").build(), 9, 10
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample,, \u201ctext").build(), 9, 10
    )
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(
        TextSegment.Builder().set_text("sample, a \u201ctext").build(), 10, 11
    )
    assert not quotation_mark_string_match.has_quote_introducer_in_leading_substring()

    quotation_mark_string_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("sample, text").build(), 8, 9)
    assert quotation_mark_string_match.has_quote_introducer_in_leading_substring()
