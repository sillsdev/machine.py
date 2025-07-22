from typing import Generator, List, Set, Union

from machine.corpora import (
    QuotationMarkUpdateSettings,
    QuotationMarkUpdateStrategy,
    QuoteConventionChangingUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmParserHandler,
    UsfmToken,
    UsfmTokenType,
    UsfmUpdateBlock,
    UsfmUpdateBlockElement,
    UsfmUpdateBlockElementType,
    parse_usfm,
)
from machine.corpora.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuotationMarkDirection,
    QuotationMarkFinder,
    QuotationMarkMetadata,
    QuotationMarkResolutionIssue,
    QuotationMarkResolver,
    QuotationMarkStringMatch,
    QuoteConventionSet,
    TextSegment,
    UsfmMarkerType,
)


def test_quotes_spanning_verses() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really said,
    \\v 2 “You shall not eat of any tree of the garden”?»
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, \n"
        + "\\v 2 ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(input_usfm, "western_european", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_single_embed() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    \\f + \\ft «This is a “footnote”» \\f*
    of the field which Yahweh God had made.
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal "
        + "\\f + \\ft “This is a ‘footnote’” \\f* of the field which Yahweh God had made."
    )

    observed_usfm = change_quotation_marks(input_usfm, "western_european", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_multiple_embeds() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    \\f + \\ft «This is a “footnote”» \\f*
    of the field \\f + \\ft Second «footnote» here \\f* which Yahweh God had made.
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal "
        + "\\f + \\ft “This is a ‘footnote’” \\f* of the field \\f + \\ft Second "
        + "“footnote” here \\f* which Yahweh God had made."
    )

    observed_usfm = change_quotation_marks(input_usfm, "western_european", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_quotes_in_text_and_embed() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God really \\f + \\ft a
    «footnote» in the «midst of “text”» \\f* said,
    “You shall not eat of any tree of the garden”?»
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really \\f + \\ft a “footnote” in the “midst of ‘text’” \\f* "
        + "said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(input_usfm, "western_european", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_quotes_in_multiple_verses_and_embed() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, «Has God
    \\v 2 really \\f + \\ft a
    «footnote» in the «midst of “text”» \\f* said,
    “You shall not eat of any tree of the garden”?»
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God\n"
        + "\\v 2 really \\f + \\ft a “footnote” in the “midst of ‘text’” \\f* "
        + "said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(input_usfm, "western_european", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


# Fallback mode does not consider the nesting of quotation marks,
# but only determines opening/closing marks and maps based on that.
def test_fallback_strategy_same_as_full() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ‘Has God really said,
    “You shall not eat of any tree of the garden”?’
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, ‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "british_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_strategy_incorrectly_nested() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ‘Has God really said,
    ‘You shall not eat of any tree of the garden’?’
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, “You shall not eat of any tree of the garden”?”"
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "british_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_strategy_incorrectly_nested_second_case() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?’
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, ‘Has God really said, “You shall not eat of any tree of the garden”?”"
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "british_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_fallback_strategy_unclosed_quote() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, ‘Has God really said,
    You shall not eat of any tree of the garden”?’
    """
    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "british_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_default_quotation_mark_update_strategy() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_skipped_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FULL),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.APPLY_FALLBACK),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(default_chapter_strategy=QuotationMarkUpdateStrategy.SKIP),
    )
    assert_usfm_equal(observed_usfm, expected_skipped_usfm)


def test_single_chapter_quotation_mark_update_strategy() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_basic_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_skipped_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + 'the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FULL]),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FALLBACK]),
    )
    assert_usfm_equal(observed_usfm, expected_basic_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(chapter_strategies=[QuotationMarkUpdateStrategy.SKIP]),
    )
    assert_usfm_equal(observed_usfm, expected_skipped_usfm)


def test_multiple_chapter_same_strategy() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle" than any animal
    of the field which Yahweh God had made.
    \\c 2
    \\v 1 He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_usfm = (
        "\\c 1\n"
        + '\\v 1 Now the serpent was more subtle" than any animal of the field which Yahweh God had made.\n'
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_fallback_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(
            chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FULL, QuotationMarkUpdateStrategy.APPLY_FULL]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_full_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(
            chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.APPLY_FALLBACK]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_fallback_usfm)


def test_multiple_chapter_multiple_strategies() -> None:
    normalized_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle" than any animal
    of the field which Yahweh God had made.
    \\c 2
    \\v 1 He said to the woman, "Has God really said,
    You shall not eat of any tree of the garden'?"
    """
    expected_full_then_fallback_usfm = (
        "\\c 1\n"
        + '\\v 1 Now the serpent was more subtle" than any animal of the field which Yahweh God had made.\n'
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden’?”"
    )

    expected_fallback_then_full_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + "\\v 1 He said to the woman, “Has God really said, You shall not eat of any tree of the garden'?”"
    )

    expected_fallback_then_skip_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle” than any animal of the field which Yahweh God had made.\n"
        + "\\c 2\n"
        + '\\v 1 He said to the woman, "Has God really said, You shall not eat of any tree of the garden\'?"'
    )

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(
            chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FULL, QuotationMarkUpdateStrategy.APPLY_FALLBACK]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_full_then_fallback_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(
            chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.APPLY_FULL]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_fallback_then_full_usfm)

    observed_usfm = change_quotation_marks(
        normalized_usfm,
        "typewriter_english",
        "standard_english",
        QuotationMarkUpdateSettings(
            chapter_strategies=[QuotationMarkUpdateStrategy.APPLY_FALLBACK, QuotationMarkUpdateStrategy.SKIP]
        ),
    )
    assert_usfm_equal(observed_usfm, expected_fallback_then_skip_usfm)


def test_multi_character_quotation_marks_in_source_quote_convention() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, <<Has God really said,
    <You shall not eat of any tree of the garden>?>>
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, “Has God really said, "
        + "‘You shall not eat of any tree of the garden’?”"
    )

    observed_usfm = change_quotation_marks(input_usfm, "typewriter_french", "standard_english")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_multi_character_quotation_marks_in_target_quote_convention() -> None:
    input_usfm = """\\c 1
    \\v 1 Now the serpent was more subtle than any animal
    of the field which Yahweh God had made.
    He said to the woman, “Has God really said,
    ‘You shall not eat of any tree of the garden’?”
    """

    expected_usfm = (
        "\\c 1\n"
        + "\\v 1 Now the serpent was more subtle than any animal of the field which Yahweh God had made. He said to "
        + "the woman, <<Has God really said, "
        + "<You shall not eat of any tree of the garden>?>>"
    )

    observed_usfm = change_quotation_marks(input_usfm, "standard_english", "typewriter_french")
    assert_usfm_equal(observed_usfm, expected_usfm)


def test_process_scripture_element() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "british_english")
    )
    quote_convention_changer._quotation_mark_finder = MockQuotationMarkFinder()

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[UsfmToken(UsfmTokenType.TEXT, text="test segment")],
    )
    mock_quotation_mark_resolver: QuotationMarkResolver = MockQuotationMarkResolver()
    quote_convention_changer._process_scripture_element(update_element, mock_quotation_mark_resolver)

    assert quote_convention_changer._quotation_mark_finder.num_times_called == 1
    assert mock_quotation_mark_resolver.num_times_called == 1
    assert quote_convention_changer._quotation_mark_finder.matches_to_return[0]._text_segment._text == "this is a ‘test"
    assert (
        quote_convention_changer._quotation_mark_finder.matches_to_return[1]._text_segment._text
        == "the test ends” here"
    )


def test_create_text_segments_basic() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT, tokens=[UsfmToken(UsfmTokenType.TEXT, text="test segment")]
    )
    text_segments: List[TextSegment] = quote_convention_changer._create_text_segments(update_element)

    assert len(text_segments) == 1
    assert text_segments[0]._text == "test segment"
    assert text_segments[0]._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert text_segments[0]._markers_in_preceding_context == set()
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment is None


def test_create_text_segments_with_preceding_markers() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.PARAGRAPH),
            UsfmToken(UsfmTokenType.TEXT, text="test segment"),
        ],
    )
    text_segments: List[TextSegment] = quote_convention_changer._create_text_segments(update_element)

    assert len(text_segments) == 1
    assert text_segments[0]._text == "test segment"
    assert text_segments[0]._immediate_preceding_marker == UsfmMarkerType.PARAGRAPH
    assert text_segments[0]._markers_in_preceding_context == {
        UsfmMarkerType.VERSE,
        UsfmMarkerType.PARAGRAPH,
    }
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment is None


def test_create_text_segments_with_multiple_text_tokens() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    update_element: UsfmUpdateBlockElement = UsfmUpdateBlockElement(
        UsfmUpdateBlockElementType.TEXT,
        tokens=[
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.PARAGRAPH),
            UsfmToken(UsfmTokenType.TEXT, text="test segment1"),
            UsfmToken(UsfmTokenType.VERSE),
            UsfmToken(UsfmTokenType.CHARACTER),
            UsfmToken(UsfmTokenType.TEXT, text="test segment2"),
            UsfmToken(UsfmTokenType.PARAGRAPH),
        ],
    )
    text_segments: List[TextSegment] = quote_convention_changer._create_text_segments(update_element)

    assert len(text_segments) == 2
    assert text_segments[0]._text == "test segment1"
    assert text_segments[0]._immediate_preceding_marker == UsfmMarkerType.PARAGRAPH
    assert text_segments[0]._markers_in_preceding_context == {
        UsfmMarkerType.VERSE,
        UsfmMarkerType.PARAGRAPH,
    }
    assert text_segments[0].previous_segment is None
    assert text_segments[0].next_segment == text_segments[1]
    assert text_segments[1]._text == "test segment2"
    assert text_segments[1]._immediate_preceding_marker == UsfmMarkerType.CHARACTER
    assert text_segments[1]._markers_in_preceding_context == {
        UsfmMarkerType.VERSE,
        UsfmMarkerType.CHARACTER,
    }
    assert text_segments[1].previous_segment == text_segments[0]
    assert text_segments[1].next_segment is None


def test_create_text_segment() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    usfm_token: UsfmToken = UsfmToken(UsfmTokenType.TEXT, text="test segment")
    segment: Union[TextSegment, None] = quote_convention_changer._create_text_segment(usfm_token)

    assert segment is not None
    assert segment._text == "test segment"
    assert segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert segment._markers_in_preceding_context == set()
    assert segment._usfm_token == usfm_token


def test_set_previous_and_next_for_segments() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    segments: List[TextSegment] = [
        TextSegment.Builder().set_text("segment 1 text").build(),
        TextSegment.Builder().set_text("segment 2 text").build(),
        TextSegment.Builder().set_text("segment 3 text").build(),
    ]

    quote_convention_changer._set_previous_and_next_for_segments(segments)

    assert segments[0].previous_segment is None
    assert segments[0].next_segment == segments[1]
    assert segments[1].previous_segment == segments[0]
    assert segments[1].next_segment == segments[2]
    assert segments[2].previous_segment == segments[1]
    assert segments[2].next_segment is None


def test_update_quotation_marks() -> None:
    multi_char_to_single_char_quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("typewriter_french", "standard_english")
    )

    multi_character_text_segment: TextSegment = TextSegment.Builder().set_text("this <<is <a test segment> >>").build()
    multi_character_quotation_marks: List[QuotationMarkMetadata] = [
        QuotationMarkMetadata(
            quotation_mark="<<",
            depth=1,
            direction=QuotationMarkDirection.OPENING,
            text_segment=multi_character_text_segment,
            start_index=5,
            end_index=7,
        ),
        QuotationMarkMetadata(
            quotation_mark="<",
            depth=2,
            direction=QuotationMarkDirection.OPENING,
            text_segment=multi_character_text_segment,
            start_index=10,
            end_index=11,
        ),
        QuotationMarkMetadata(
            quotation_mark=">",
            depth=2,
            direction=QuotationMarkDirection.CLOSING,
            text_segment=multi_character_text_segment,
            start_index=25,
            end_index=26,
        ),
        QuotationMarkMetadata(
            quotation_mark=">>",
            depth=1,
            direction=QuotationMarkDirection.CLOSING,
            text_segment=multi_character_text_segment,
            start_index=27,
            end_index=29,
        ),
    ]

    multi_char_to_single_char_quote_convention_changer._update_quotation_marks(multi_character_quotation_marks)

    assert multi_character_text_segment.text == "this “is ‘a test segment’ ”"

    assert multi_character_quotation_marks[0].start_index == 5
    assert multi_character_quotation_marks[0].end_index == 6
    assert multi_character_quotation_marks[0].text_segment == multi_character_text_segment

    assert multi_character_quotation_marks[1].start_index == 9
    assert multi_character_quotation_marks[1].end_index == 10
    assert multi_character_quotation_marks[1].text_segment == multi_character_text_segment

    assert multi_character_quotation_marks[2].start_index == 24
    assert multi_character_quotation_marks[2].end_index == 25
    assert multi_character_quotation_marks[2].text_segment == multi_character_text_segment

    assert multi_character_quotation_marks[3].start_index == 26
    assert multi_character_quotation_marks[3].end_index == 27
    assert multi_character_quotation_marks[3].text_segment == multi_character_text_segment

    single_char_to_multi_char_quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "typewriter_french")
    )

    single_character_text_segment: TextSegment = TextSegment.Builder().set_text("this “is ‘a test segment’ ”").build()
    single_character_quotation_marks: List[QuotationMarkMetadata] = [
        QuotationMarkMetadata(
            quotation_mark="“",
            depth=1,
            direction=QuotationMarkDirection.OPENING,
            text_segment=single_character_text_segment,
            start_index=5,
            end_index=6,
        ),
        QuotationMarkMetadata(
            quotation_mark="‘",
            depth=2,
            direction=QuotationMarkDirection.OPENING,
            text_segment=single_character_text_segment,
            start_index=9,
            end_index=10,
        ),
        QuotationMarkMetadata(
            quotation_mark="’",
            depth=2,
            direction=QuotationMarkDirection.CLOSING,
            text_segment=single_character_text_segment,
            start_index=24,
            end_index=25,
        ),
        QuotationMarkMetadata(
            quotation_mark="”",
            depth=1,
            direction=QuotationMarkDirection.CLOSING,
            text_segment=single_character_text_segment,
            start_index=26,
            end_index=27,
        ),
    ]

    single_char_to_multi_char_quote_convention_changer._update_quotation_marks(single_character_quotation_marks)

    assert single_character_text_segment.text == "this <<is <a test segment> >>"

    assert single_character_quotation_marks[0].start_index == 5
    assert single_character_quotation_marks[0].end_index == 7
    assert single_character_quotation_marks[0].text_segment == single_character_text_segment

    assert single_character_quotation_marks[1].start_index == 10
    assert single_character_quotation_marks[1].end_index == 11
    assert single_character_quotation_marks[1].text_segment == single_character_text_segment

    assert single_character_quotation_marks[2].start_index == 25
    assert single_character_quotation_marks[2].end_index == 26
    assert single_character_quotation_marks[2].text_segment == single_character_text_segment

    assert single_character_quotation_marks[3].start_index == 27
    assert single_character_quotation_marks[3].end_index == 29
    assert single_character_quotation_marks[3].text_segment == single_character_text_segment


def test_check_for_chapter_change() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler("standard_english", "standard_english")
    )

    assert quote_convention_changer._current_chapter_number == 0

    quote_convention_changer._check_for_chapter_change(UsfmUpdateBlock([ScriptureRef.parse("MAT 1:1")], []))

    assert quote_convention_changer._current_chapter_number == 1

    quote_convention_changer._check_for_chapter_change(UsfmUpdateBlock([ScriptureRef.parse("ISA 15:22")], []))

    assert quote_convention_changer._current_chapter_number == 15


def test_start_new_chapter() -> None:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler(
            "standard_english",
            "standard_english",
            QuotationMarkUpdateSettings(
                chapter_strategies=[
                    QuotationMarkUpdateStrategy.SKIP,
                    QuotationMarkUpdateStrategy.APPLY_FULL,
                    QuotationMarkUpdateStrategy.APPLY_FALLBACK,
                ]
            ),
        )
    )

    quote_convention_changer._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED).set_text(
        "this text should be erased"
    )
    quote_convention_changer._verse_text_quotation_mark_resolver._issues.add(
        QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK
    )

    quote_convention_changer._start_new_chapter(1)
    segment = quote_convention_changer._next_scripture_text_segment_builder.build()
    assert quote_convention_changer._current_strategy == QuotationMarkUpdateStrategy.SKIP
    assert segment._immediate_preceding_marker == UsfmMarkerType.CHAPTER
    assert segment._text == ""
    assert UsfmMarkerType.EMBED not in segment._markers_in_preceding_context
    assert quote_convention_changer._verse_text_quotation_mark_resolver._issues == set()

    quote_convention_changer._start_new_chapter(2)
    assert quote_convention_changer._current_strategy == QuotationMarkUpdateStrategy.APPLY_FULL

    quote_convention_changer._start_new_chapter(3)
    assert quote_convention_changer._current_strategy == QuotationMarkUpdateStrategy.APPLY_FALLBACK


def change_quotation_marks(
    normalized_usfm: str,
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_mark_update_settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
) -> str:
    quote_convention_changer: QuoteConventionChangingUsfmUpdateBlockHandler = (
        create_quote_convention_changing_usfm_update_block_handler(
            source_quote_convention_name, target_quote_convention_name, quotation_mark_update_settings
        )
    )

    updater = UpdateUsfmParserHandler(update_block_handlers=[quote_convention_changer])
    parse_usfm(normalized_usfm, updater)

    return updater.get_usfm()


def create_quote_convention_changing_usfm_update_block_handler(
    source_quote_convention_name: str,
    target_quote_convention_name: str,
    quotation_mark_update_settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
) -> QuoteConventionChangingUsfmUpdateBlockHandler:
    source_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(source_quote_convention_name)
    assert source_quote_convention is not None

    target_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(target_quote_convention_name)
    assert target_quote_convention is not None

    return QuoteConventionChangingUsfmUpdateBlockHandler(
        source_quote_convention,
        target_quote_convention,
        quotation_mark_update_settings,
    )


def assert_usfm_equal(observed_usfm: str, expected_usfm: str) -> None:
    for observed_line, expected_line in zip(observed_usfm.split("\n"), expected_usfm.split("\n")):
        assert observed_line.strip() == expected_line.strip()


class MockQuotationMarkFinder(QuotationMarkFinder):
    def __init__(self) -> None:
        super().__init__(QuoteConventionSet([]))
        self.num_times_called = 0
        self.matches_to_return = [
            QuotationMarkStringMatch(TextSegment.Builder().set_text('this is a "test').build(), 10, 11),
            QuotationMarkStringMatch(TextSegment.Builder().set_text('the test ends" here').build(), 13, 14),
        ]

    def find_all_potential_quotation_marks_in_text_segments(
        self, text_segments: List[TextSegment]
    ) -> List[QuotationMarkStringMatch]:
        self.num_times_called += 1
        return self.matches_to_return


class MockQuotationMarkResolver(QuotationMarkResolver):
    def __init__(self):
        self.num_times_called = 0

    def reset(self) -> None:
        self.num_times_called = 0

    def resolve_quotation_marks(
        self, quote_matches: List[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        self.num_times_called += 1
        current_depth = 1
        current_direction = QuotationMarkDirection.OPENING
        for quote_match in quote_matches:
            yield quote_match.resolve(current_depth, current_direction)
            current_depth += 1
            current_direction = (
                QuotationMarkDirection.CLOSING
                if current_direction == QuotationMarkDirection.OPENING
                else QuotationMarkDirection.OPENING
            )

    def get_issues(self) -> Set[QuotationMarkResolutionIssue]:
        return set()
