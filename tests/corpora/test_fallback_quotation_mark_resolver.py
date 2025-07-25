from machine.corpora import FallbackQuotationMarkResolver, QuotationMarkUpdateResolutionSettings
from machine.corpora.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuotationMarkResolutionIssue,
    QuotationMarkStringMatch,
    QuoteConventionDetectionResolutionSettings,
    QuoteConventionSet,
    TextSegment,
)


def test_reset():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention)
    )

    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"\'test text"').build(), 0, 1
    )
    basic_quotation_mark_resolver._issues.add(QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK)

    basic_quotation_mark_resolver.reset()
    assert basic_quotation_mark_resolver._last_quotation_mark is None
    assert len(basic_quotation_mark_resolver._issues) == 0


def test_simple_quotation_mark_resolution_with_no_previous_mark():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    actual_resolved_quotation_marks = list(
        basic_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(TextSegment.Builder().set_text('test " text').build(), 5, 6),
            ]
        )
    )
    expected_resolved_quotation_marks = [
        QuotationMarkMetadata(
            '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('test " text').build(), 5, 6
        ),
    ]

    assert_resolved_quotation_marks_equal(
        actual_resolved_quotation_marks,
        expected_resolved_quotation_marks,
    )


def test_simple_quotation_mark_resolution_with_previous_opening_mark():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    actual_resolved_quotation_marks = list(
        basic_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(TextSegment.Builder().set_text('"test " text').build(), 0, 1),
                QuotationMarkStringMatch(TextSegment.Builder().set_text('"test " text').build(), 6, 7),
            ]
        )
    )
    expected_resolved_quotation_marks = [
        QuotationMarkMetadata(
            '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"test " text').build(), 0, 1
        ),
        QuotationMarkMetadata(
            '"', 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().set_text('"test " text').build(), 6, 7
        ),
    ]

    assert_resolved_quotation_marks_equal(
        actual_resolved_quotation_marks,
        expected_resolved_quotation_marks,
    )


def test_simple_quotation_mark_resolution_with_previous_closing_mark():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    actual_resolved_quotation_marks = list(
        basic_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(TextSegment.Builder().set_text('test" " text').build(), 4, 5),
                QuotationMarkStringMatch(TextSegment.Builder().set_text('test" " text').build(), 6, 7),
            ]
        )
    )
    expected_resolved_quotation_marks = [
        QuotationMarkMetadata(
            '"', 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().set_text('test" " text').build(), 4, 5
        ),
        QuotationMarkMetadata(
            '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('test" " text').build(), 6, 7
        ),
    ]

    assert_resolved_quotation_marks_equal(
        actual_resolved_quotation_marks,
        expected_resolved_quotation_marks,
    )


def test_is_opening_quote():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    # valid opening quote at start of segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 0, 1)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # opening quote with leading whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test "text"').build(), 5, 6)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # opening quote with quote introducer
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test:"text"').build(), 5, 6)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # QuotationMarkStringMatch indices don't indicate a quotation mark
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test "text"').build(), 0, 1)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False

    # the quotation mark is not valid under the current quote convention
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('<test "text"').build(), 0, 1)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False

    # no leading whitespace before quotation mark
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test"text"').build(), 4, 5)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False

    # closing quote at the end of the segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 10, 11)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False

    # closing quote with trailing whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text" ').build(), 10, 11)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False


def test_is_opening_quote_with_unambiguous_quote_convention():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuoteConventionDetectionResolutionSettings(QuoteConventionSet([english_quote_convention]))
    )

    # unambiguous opening quote at start of segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("“test text”").build(), 0, 1)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # unambiguous opening quote with leading whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("test “text”").build(), 5, 6)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # unambiguous opening quote without the "correct" context
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("test“text”").build(), 4, 5)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True

    # unambiguous closing quote
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("“test” text").build(), 5, 6)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False


def test_is_opening_quote_stateful():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    # no preceding quote
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"\'test text"').build(), 1, 2)
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is False

    # immediately preceding quote
    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"\'test text"').build(), 0, 1
    )
    assert basic_quotation_mark_resolver._is_opening_quotation_mark(quote_match) is True


def test_does_most_recent_opening_mark_immediately_precede():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention)
    )

    # no preceding quote
    nested_quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"\'test text"').build(), 1, 2)
    assert basic_quotation_mark_resolver._does_most_recent_opening_mark_immediately_precede(nested_quote_match) is False

    # correct preceding quote
    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"\'test text"').build(), 0, 1
    )
    assert basic_quotation_mark_resolver._does_most_recent_opening_mark_immediately_precede(nested_quote_match) is True

    # wrong direction for preceding quote
    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().set_text('"\'test text"').build(), 0, 1
    )
    assert basic_quotation_mark_resolver._does_most_recent_opening_mark_immediately_precede(nested_quote_match) is False

    # different text segment for preceding quote
    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"\'different text"').build(), 0, 1
    )
    assert basic_quotation_mark_resolver._does_most_recent_opening_mark_immediately_precede(nested_quote_match) is False

    # previous quote is not *immediately* before the current quote
    nested_quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('" \'test text"').build(), 2, 3)
    basic_quotation_mark_resolver._last_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('" \'test text"').build(), 0, 1
    )
    assert basic_quotation_mark_resolver._does_most_recent_opening_mark_immediately_precede(nested_quote_match) is False


def test_is_closing_quote():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    # valid closing quote at end of segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 10, 11)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # closing quote with trailing whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test" text').build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # closing quote with trailing punctuation
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text".').build(), 10, 11)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # QuotationMarkStringMatch indices don't indicate a quotation mark
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 9, 10)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False

    # the quotation mark is not valid under the current quote convention
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test "text>').build(), 10, 11)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False

    # no trailing whitespace after quotation mark
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test"text').build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False

    # opening quote at the start of the segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 0, 1)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False

    # opening quote with leading whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text('test "text"').build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False


def test_is_closing_quote_with_unambiguous_quote_convention():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuoteConventionDetectionResolutionSettings(QuoteConventionSet([english_quote_convention]))
    )

    # unambiguous closing quote at end of segment
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("“test text”").build(), 10, 11)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # unambiguous closing quote with trailing whitespace
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("“test” text").build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # unambiguous closing quote without the "correct" context
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("“test”text").build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is True

    # unambiguous opening quote
    quote_match = QuotationMarkStringMatch(TextSegment.Builder().set_text("test “text”").build(), 5, 6)
    assert basic_quotation_mark_resolver._is_closing_quotation_mark(quote_match) is False


def test_resolve_opening_quote():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    expected_resolved_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text('"test text"').build(), 0, 1
    )
    actual_resolved_quotation_mark = basic_quotation_mark_resolver._resolve_opening_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 0, 1)
    )
    assert actual_resolved_quotation_mark == expected_resolved_quotation_mark
    assert basic_quotation_mark_resolver._last_quotation_mark == actual_resolved_quotation_mark


def test_resolve_closing_quote():
    english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert english_quote_convention is not None

    basic_quotation_mark_resolver = FallbackQuotationMarkResolver(
        QuotationMarkUpdateResolutionSettings(english_quote_convention.normalize())
    )

    expected_resolved_quotation_mark = QuotationMarkMetadata(
        '"', 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().set_text('"test text"').build(), 10, 11
    )
    actual_resolved_quotation_mark = basic_quotation_mark_resolver._resolve_closing_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"test text"').build(), 10, 11)
    )
    assert actual_resolved_quotation_mark == expected_resolved_quotation_mark


def assert_resolved_quotation_marks_equal(
    actual_resolved_quotation_marks: list[QuotationMarkMetadata],
    expected_resolved_quotation_marks: list[QuotationMarkMetadata],
) -> None:
    assert len(actual_resolved_quotation_marks) == len(expected_resolved_quotation_marks)
    for actual_mark, expected_mark in zip(actual_resolved_quotation_marks, expected_resolved_quotation_marks):
        assert actual_mark == expected_mark
