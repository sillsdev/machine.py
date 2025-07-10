from pytest import raises

from machine.corpora import QuotationMarkUpdateResolutionSettings
from machine.corpora.punctuation_analysis import (
    DepthBasedQuotationMarkResolver,
    QuotationMarkCategorizer,
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuotationMarkResolutionIssue,
    QuotationMarkResolverState,
    QuotationMarkStringMatch,
    QuoteContinuerState,
    QuoteContinuerStyle,
    QuoteConventionDetectionResolutionSettings,
    QuoteConventionSet,
    TextSegment,
    UsfmMarkerType,
    standard_quote_conventions,
)


# QuotationMarkResolverState tests
def test_current_depth_quotation_mark_resolver_state() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    assert quotation_mark_resolver_state.current_depth == 0

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.current_depth == 1

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.current_depth == 2

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.current_depth == 1

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.current_depth == 0


def test_has_open_quotation_mark() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    assert not quotation_mark_resolver_state.has_open_quotation_mark()

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.has_open_quotation_mark()

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.has_open_quotation_mark()

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.has_open_quotation_mark()

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not quotation_mark_resolver_state.has_open_quotation_mark()


def test_are_more_than_n_quotes_open() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(1)
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(2)

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(1)
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(2)

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.are_more_than_n_quotes_open(1)
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(2)

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(1)
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(2)

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(1)
    assert not quotation_mark_resolver_state.are_more_than_n_quotes_open(2)


def test_get_opening_quotation_mark_at_depth() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    with raises(Exception):
        quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(1)

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(1) == "\u201c"
    with raises(Exception):
        quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(2)

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(1) == "\u201c"
    assert quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(2) == "\u2018"

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(1) == "\u201c"
    with raises(Exception):
        quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(2)

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    with raises(Exception):
        quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(1)


def test_get_deepest_opening_mark() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    with raises(Exception):
        quotation_mark_resolver_state.get_deepest_opening_quotation_mark()

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == "\u201c"

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == "\u2018"

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == "\u201c"

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    with raises(Exception):
        quotation_mark_resolver_state.get_deepest_opening_quotation_mark()


# QuotationContinuerState tests
def test_get_current_depth_quotation_continuer_state() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )

    quotation_continuer_state = QuoteContinuerState()
    assert quotation_continuer_state.current_depth == 0

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.current_depth == 1

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.current_depth == 2

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.current_depth == 0


def test_has_continuer_been_observed() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )

    quotation_continuer_state = QuoteContinuerState()
    assert not quotation_continuer_state.continuer_has_been_observed()

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.continuer_has_been_observed()

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.continuer_has_been_observed()

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert not quotation_continuer_state.continuer_has_been_observed()


def test_get_continuer_style() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )

    quotation_continuer_state = QuoteContinuerState()
    assert quotation_continuer_state.continuer_style is QuoteContinuerStyle.UNDETERMINED

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.continuer_style is QuoteContinuerStyle.ENGLISH

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.SPANISH,
    )
    assert quotation_continuer_state.continuer_style is QuoteContinuerStyle.SPANISH

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert quotation_continuer_state.continuer_style is QuoteContinuerStyle.ENGLISH


def test_add_quotation_continuer() -> None:
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )

    quotation_continuer_state = QuoteContinuerState()

    assert quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    ) == QuotationMarkMetadata(
        "\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text("\u201c").build(), 0, 1
    )

    assert quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.SPANISH,
    ) == QuotationMarkMetadata(
        "\u2018", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text("\u2018").build(), 0, 1
    )
    assert quotation_continuer_state.continuer_style == QuoteContinuerStyle.SPANISH

    assert quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    ) == QuotationMarkMetadata(
        "\u201c", 3, QuotationMarkDirection.OPENING, TextSegment.Builder().set_text("\u201c").build(), 0, 1
    )


# QuotationMarkCategorizer tests


def test_is_english_quotation_continuer() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None

    english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()

    quotation_mark_categorizer = QuotationMarkCategorizer(
        english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )

    # Should always be false if the continuer style is Spanish
    quotation_continuer_state._continuer_style = QuoteContinuerStyle.ENGLISH
    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    quotation_continuer_state._continuer_style = QuoteContinuerStyle.SPANISH
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )
    quotation_continuer_state._continuer_style = QuoteContinuerStyle.ENGLISH

    # Should be false if there's no preceding paragraph marker (and the settings say to rely on markers)
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").build(),
            0,
            1,
        ),
        None,
        None,
    )

    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    quotation_mark_categorizer_for_denormalization = QuotationMarkCategorizer(
        QuotationMarkUpdateResolutionSettings(standard_english_quote_convention, standard_english_quote_convention),
        quotation_mark_resolver_state,
        quotation_continuer_state,
    )
    assert quotation_mark_categorizer_for_denormalization.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").build(),
            0,
            1,
        ),
        None,
        None,
    )

    # Should be false if there are no open quotation marks
    empty_quotation_mark_resolver_state = QuotationMarkResolverState()
    empty_quotation_mark_categorizer = QuotationMarkCategorizer(
        english_resolver_settings, empty_quotation_mark_resolver_state, quotation_continuer_state
    )
    assert not empty_quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    # Should be false if the starting index of the quotation mark is greater than 0
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text(" \u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    # Should be false if the mark does not match the already opened mark
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u2018test").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    # If there are multiple open quotes, the next quote continuer must follow immediately
    # after the current one
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )
    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u2018test").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u2018test").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
    )
    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
    )

    # When there are multiple open quotes, the continuer must match the deepest observed mark
    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u2018test").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )

    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c\u2018test").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201c").build(),
            0,
            1,
        )
    )

    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u201c\u2018\u201ctest")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            1,
            2,
        ),
        None,
        None,
    )

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u201c\u2018\u201ctest")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            1,
            2,
        ),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.ENGLISH,
    )
    assert not quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u201c\u2018\u2018test")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            2,
            3,
        ),
        None,
        None,
    )
    assert quotation_mark_categorizer.is_english_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u201c\u2018\u201ctest")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            2,
            3,
        ),
        None,
        None,
    )


def test_is_spanish_quotation_continuer() -> None:
    western_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("western_european")
    )
    assert western_european_quote_convention is not None

    spanish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([western_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()

    quotation_mark_categorizer = QuotationMarkCategorizer(
        spanish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00ab").build(), 0, 1)
    )

    # Should always be false if the continuer style is English
    quotation_continuer_state._continuer_style = QuoteContinuerStyle.SPANISH
    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    quotation_continuer_state._continuer_style = QuoteContinuerStyle.ENGLISH
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )
    quotation_continuer_state._continuer_style = QuoteContinuerStyle.SPANISH

    # Should be false if there's no preceding paragraph marker (and the settings say to rely on markers)
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").build(),
            0,
            1,
        ),
        None,
        None,
    )

    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    quotation_mark_categorizer_for_denormalization = QuotationMarkCategorizer(
        QuotationMarkUpdateResolutionSettings(western_european_quote_convention, western_european_quote_convention),
        quotation_mark_resolver_state,
        quotation_continuer_state,
    )
    assert quotation_mark_categorizer_for_denormalization.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").build(),
            0,
            1,
        ),
        None,
        None,
    )

    # Should be false if there are no open quotation marks
    empty_quotation_mark_resolver_state = QuotationMarkResolverState()
    empty_quotation_mark_categorizer = QuotationMarkCategorizer(
        spanish_resolver_settings, empty_quotation_mark_resolver_state, quotation_continuer_state
    )
    assert not empty_quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    # Should be false if the starting index of the quotation mark is greater than 0
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text(" \u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    # Should be false if the mark does not match the already opened mark
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u201dtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )

    # If there are multiple open quotes, the next quote continuer must follow immediately
    # after the current one
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        None,
    )
    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u201dtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u201dtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
    )
    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        None,
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u00bbtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
    )

    # When there are multiple open quotes, the continuer must match the deepest observed mark
    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u201dtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.SPANISH,
    )

    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u201ctest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u00bb\u201dtest").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            1,
            2,
        ),
        None,
        None,
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("\u2018").build(),
            0,
            1,
        )
    )

    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u00bb\u201d\u2019test")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            1,
            2,
        ),
        None,
        None,
    )

    quotation_continuer_state.add_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u00bb\u201d\u2019test")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            1,
            2,
        ),
        quotation_mark_resolver_state,
        QuoteContinuerStyle.SPANISH,
    )
    assert not quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u00bb\u201d\u201dtest")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            2,
            3,
        ),
        None,
        None,
    )
    assert quotation_mark_categorizer.is_spanish_quote_continuer(
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("\u00bb\u201d\u2019test")
            .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            .build(),
            2,
            3,
        ),
        None,
        None,
    )


def test_is_opening_quote() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    central_european_quotation_mark_categorizer = QuotationMarkCategorizer(
        central_european_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    british_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("british_english")
    )
    assert british_english_quote_convention is not None
    british_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([british_english_quote_convention])
    )
    british_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        british_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_swedish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    three_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [central_european_quote_convention, british_english_quote_convention, standard_swedish_quote_convention]
        )
    )
    three_conventions_quotation_mark_categorizer = QuotationMarkCategorizer(
        three_conventions_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # It should only accept valid opening marks under the quote convention
    assert central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e").build(), 1, 2)
    )
    assert central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' "').build(), 1, 2)
    )

    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a").build(), 1, 2)
    )
    assert british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c").build(), 1, 2)
    )
    assert british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' "').build(), 1, 2)
    )

    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' "').build(), 1, 2)
    )

    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' "').build(), 1, 2)
    )

    # Leading whitespace is not necessary for unambiguous opening quotes
    assert central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u201e").build(), 4, 5)
    )
    assert central_european_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u201a").build(), 4, 5)
    )
    assert british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u201c").build(), 4, 5)
    )
    assert british_english_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u2018").build(), 4, 5)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u201e").build(), 4, 5)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("text\u201a").build(), 4, 5)
    )

    # An ambiguous quotation mark (opening/closing) is recognized as opening if
    # it has a quote introducer beforehand
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(",\u201d").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(":\u2019").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(",\u201c").build(), 1, 2)
    )

    # An ambiguous quotation mark (opening/closing) is recognized as opening if
    # preceded by another opening mark
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201d").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2019").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u201c").build(), 1, 2)
    )

    # An ambiguous quotation mark (opening/closing) is not recognized as opening if
    # it has trailing whitespace or punctuation
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d.").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(",\u201d ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2019 ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c\u2019?").build(), 1, 2)
    )


def test_is_closing_quote() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    central_european_quotation_mark_categorizer = QuotationMarkCategorizer(
        central_european_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    british_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("british_english")
    )
    assert british_english_quote_convention is not None
    british_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([british_english_quote_convention])
    )
    british_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        british_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_swedish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_french_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_french")
    )
    assert standard_french_quote_convention is not None
    standard_french_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_french_quote_convention])
    )
    standard_french_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_french_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    three_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [central_european_quote_convention, british_english_quote_convention, standard_swedish_quote_convention]
        )
    )
    three_conventions_quotation_mark_categorizer = QuotationMarkCategorizer(
        three_conventions_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # It should only accept valid closing marks under the quote convention
    assert central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c ").build(), 0, 1)
    )
    assert central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018 ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201e ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201a ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019 ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb ").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('" ').build(), 0, 1)
    )

    assert not british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c ").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018 ").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019 ").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb ").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('" ').build(), 0, 1)
    )

    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c ").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018 ").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019 ").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb ").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('" ').build(), 0, 1)
    )

    assert three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c ").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018 ").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019 ").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb ").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('" ').build(), 0, 1)
    )

    # Trailing whitespace is not necessary for unambiguous closing quotes
    assert standard_french_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bbtext").build(), 0, 1)
    )
    assert standard_french_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u203atext").build(), 0, 1)
    )

    # An ambiguous quotation mark (opening/closing) is recognized as closing if
    # followed by whitespace, punctuation or the end of the segment
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201dtext").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019text").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019?").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019\u201d").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201ctext").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c?").build(), 0, 1)
    )

    # An ambiguous quotation mark (opening/closing) is not recognized as opening if
    # it has leading whitespace
    assert not standard_swedish_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\t\u201c?").build(), 1, 2)
    )


def test_is_malformed_opening_quote() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    central_european_quotation_mark_categorizer = QuotationMarkCategorizer(
        central_european_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    british_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("british_english")
    )
    assert british_english_quote_convention is not None
    british_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([british_english_quote_convention])
    )
    british_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        british_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_swedish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    three_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [central_european_quote_convention, british_english_quote_convention, standard_swedish_quote_convention]
        )
    )
    three_conventions_quotation_mark_categorizer = QuotationMarkCategorizer(
        three_conventions_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # It should only accept valid opening marks under the quote convention
    assert central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e ").build(), 1, 2)
    )
    assert central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018 ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' " ').build(), 1, 2)
    )

    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a ").build(), 1, 2)
    )
    assert british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )
    assert british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018 ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' " ').build(), 1, 2)
    )

    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018 ").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' " ').build(), 1, 2)
    )

    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201e ").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201a ").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018 ").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u00ab ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(' " ').build(), 1, 2)
    )

    # Should return true if there is a leading quote introducer
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(",\u201d ").build(), 1, 2)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019 ").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(":\u2019 ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c ").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(",\u201c ").build(), 1, 2)
    )

    # Should return false unless the mark has leading and trailing whitespace
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )

    # Should return false if there is already an open quotation mark on the stack
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201c ").build(), 1, 2)
    )


def test_is_malformed_closing_quote() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    central_european_quotation_mark_categorizer = QuotationMarkCategorizer(
        central_european_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    british_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("british_english")
    )
    assert british_english_quote_convention is not None
    british_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([british_english_quote_convention])
    )
    british_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        british_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_swedish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    three_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [central_european_quote_convention, british_english_quote_convention, standard_swedish_quote_convention]
        )
    )
    three_conventions_quotation_mark_categorizer = QuotationMarkCategorizer(
        three_conventions_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # It should only accept valid closing marks under the quote convention
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201e").build(), 0, 1)
    )
    assert central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201e").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201a").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    assert not three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    # Returns true if it's at the end of the segment
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )

    # Returns true if it does not have trailing whitespace
    assert standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d-").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201dtext").build(), 0, 1)
    )

    # Returns true if it has trailing and leading whitespace
    assert standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d ").build(), 1, 2)
    )

    # Requires there to be an open quotation mark on the stack
    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )

    # Requires the quotation mark on the stack to be a valid pair with the
    # observed quotation mark
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_malformed_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )


def test_is_unpaired_closing_quote() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    central_european_quotation_mark_categorizer = QuotationMarkCategorizer(
        central_european_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    british_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("british_english")
    )
    assert british_english_quote_convention is not None
    british_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([british_english_quote_convention])
    )
    british_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        british_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_swedish_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    three_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [central_european_quote_convention, british_english_quote_convention, standard_swedish_quote_convention]
        )
    )
    three_conventions_quotation_mark_categorizer = QuotationMarkCategorizer(
        three_conventions_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # It should only accept valid closing marks under the quote convention
    assert central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201e").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201a").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    assert three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u00bb").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )

    # There must not be an opening quotation mark on the stack
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not central_european_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not standard_swedish_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not three_conventions_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )

    # There must not be leading whitespace
    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u201d").build(), 1, 2)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\t\u2019").build(), 1, 2)
    )

    # The quotation mark must be either at the end of the segment
    # or have trailing whitespace
    assert british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d").build(), 0, 1)
    )
    assert british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d ").build(), 0, 1)
    )
    assert not british_english_quotation_mark_categorizer.is_unpaired_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201d?").build(), 0, 1)
    )


def test_is_apostrophe() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    quotation_mark_resolver_state = QuotationMarkResolverState()
    quotation_continuer_state = QuoteContinuerState()
    standard_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        standard_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    typewriter_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_english")
    )
    assert typewriter_english_quote_convention is not None
    typewriter_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([typewriter_english_quote_convention])
    )
    typewriter_english_quotation_mark_categorizer = QuotationMarkCategorizer(
        typewriter_english_resolver_settings, quotation_mark_resolver_state, quotation_continuer_state
    )

    # The quotation mark must make for a plausible apostrophe
    assert typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a'b").build(), 1, 2), None
    )
    assert typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2019b").build(), 1, 2), None
    )
    assert typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2018b").build(), 1, 2), None
    )
    assert not typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u201cb").build(), 1, 2), None
    )
    assert not typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('a"b').build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a'b").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2019b").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2018b").build(), 1, 2), None
    )
    assert not standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u201cb").build(), 1, 2), None
    )
    assert not standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('a"b').build(), 1, 2), None
    )

    # Returns true if the mark has Latin letters on both sides
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2019Ƅ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("ǡ\u2019b").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("ᴀ\u2019Ｂ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("𝼀\u2019Ꝙ").build(), 1, 2), None
    )
    assert not standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2019ℵ").build(), 1, 2), None
    )
    assert typewriter_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("a\u2019ℵ").build(), 1, 2), None
    )

    # Recognizes s possessives (e.g. Moses')
    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2019").build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u2019 ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("Moses\u2019 ").build(), 5, 6), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u2019?").build(), 1, 2), None
    )
    assert not standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u20195").build(), 1, 2), None
    )

    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u2019 ").build(), 1, 2), None
    )

    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u2018").build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u2019 ").build(), 1, 2),
        QuotationMarkStringMatch(TextSegment.Builder().set_text("word\u2019").build(), 4, 5),
    )
    assert not standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("s\u2019 ").build(), 1, 2),
        QuotationMarkStringMatch(TextSegment.Builder().set_text("word\u201d").build(), 4, 5),
    )

    # the straight quote should always be an apostrophe if it's not a valid quotation mark
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("5'ℵ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" ' ").build(), 1, 2), None
    )

    # the straight quote should be an apostrophe if there's nothing on the quotation mark stack
    quotation_mark_resolver_state.add_closing_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text('"').build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("5'ℵ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" ' ").build(), 1, 2), None
    )

    # any matching mark should be an apostrophe if it doesn't pair with the
    # deepest opening quotation mark on the stack
    # (opening/closing quotation marks will have been detected before calling this)
    quotation_mark_resolver_state.add_opening_quotation_mark(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201c").build(), 0, 1)
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("5'ℵ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" ' ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("5\u2018ℵ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2018 ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text("5\u2019ℵ").build(), 1, 2), None
    )
    assert standard_english_quotation_mark_categorizer.is_apostrophe(
        QuotationMarkStringMatch(TextSegment.Builder().set_text(" \u2019 ").build(), 1, 2), None
    )


# DepthBasedQuotationMarkResolver tests
def test_depth_based_quotation_mark_resolver_reset() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201cThis is a quote").build(), 0, 1)]
        )
    )
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }

    standard_english_quotation_mark_resolver.reset()
    assert standard_english_quotation_mark_resolver.get_issues() == set()

    list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [QuotationMarkStringMatch(TextSegment.Builder().set_text("This is a quote\u2019").build(), 15, 16)]
        )
    )
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }


def test_basic_quotation_mark_recognition() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = TextSegment.Builder().set_text("\u201cThis is a \u2018quote\u2019\u201d").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()


def test_resolution_only_of_passed_matches() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = TextSegment.Builder().set_text("\u201cThis is a \u2018quote\u2019\u201d").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }

    text_segment = TextSegment.Builder().set_text("\u201cThis is a \u2018quote\u2019\u201d").build()
    assert (
        list(
            standard_english_quotation_mark_resolver.resolve_quotation_marks(
                [
                    QuotationMarkStringMatch(text_segment, 17, 18),
                ]
            )
        )
        == []
    )
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }


def test_resolution_across_segments() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment1 = TextSegment.Builder().set_text("\u201cThis is a ").build()
    text_segment2 = TextSegment.Builder().set_text("\u2018quote\u2019\u201d").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment1, 0, 1),
                QuotationMarkStringMatch(text_segment2, 0, 1),
                QuotationMarkStringMatch(text_segment2, 6, 7),
                QuotationMarkStringMatch(text_segment2, 7, 8),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment1, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment2, 0, 1),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment2, 6, 7),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment2, 7, 8),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()


def test_resolution_with_apostrophes() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = (
        TextSegment.Builder()
        .set_text("\u201cThis\u2019 is a \u2018quote\u2019\u201d")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 5, 6),
                QuotationMarkStringMatch(text_segment, 12, 13),
                QuotationMarkStringMatch(text_segment, 18, 19),
                QuotationMarkStringMatch(text_segment, 19, 20),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment, 12, 13),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment, 19, 20),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()

    typewriter_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_english")
    )
    assert typewriter_english_quote_convention is not None
    typewriter_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([typewriter_english_quote_convention])
    )
    typewriter_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(typewriter_english_resolver_settings)

    text_segment = (
        TextSegment.Builder().set_text("\"This' is a 'quote'\"").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build()
    )
    assert list(
        typewriter_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 5, 6),
                QuotationMarkStringMatch(text_segment, 12, 13),
                QuotationMarkStringMatch(text_segment, 18, 19),
                QuotationMarkStringMatch(text_segment, 19, 20),
            ]
        )
    ) == [
        QuotationMarkMetadata('"', 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("'", 2, QuotationMarkDirection.OPENING, text_segment, 12, 13),
        QuotationMarkMetadata("'", 2, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
        QuotationMarkMetadata('"', 1, QuotationMarkDirection.CLOSING, text_segment, 19, 20),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()


def test_english_quote_continuers() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment1 = TextSegment.Builder().set_text("\u201cThis is a \u2018quote").build()
    text_segment2 = (
        TextSegment.Builder()
        .set_text("\u201c\u2018This is the rest\u2019 of it\u201d")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment1, 0, 1),
                QuotationMarkStringMatch(text_segment1, 11, 12),
                QuotationMarkStringMatch(text_segment2, 0, 1),
                QuotationMarkStringMatch(text_segment2, 1, 2),
                QuotationMarkStringMatch(text_segment2, 18, 19),
                QuotationMarkStringMatch(text_segment2, 25, 26),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment1, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment1, 11, 12),
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment2, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment2, 1, 2),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment2, 18, 19),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment2, 25, 26),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()


def test_spanish_quote_continuers() -> None:
    western_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("western_european")
    )
    assert western_european_quote_convention is not None
    western_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([western_european_quote_convention])
    )
    western_european_quotation_mark_resolver = DepthBasedQuotationMarkResolver(western_european_resolver_settings)

    text_segment1 = TextSegment.Builder().set_text("\u00abThis is a \u201cquote").build()
    text_segment2 = (
        TextSegment.Builder()
        .set_text("\u00bb\u201dThis is the rest\u201d of it\u00bb")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        western_european_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment1, 0, 1),
                QuotationMarkStringMatch(text_segment1, 11, 12),
                QuotationMarkStringMatch(text_segment2, 0, 1),
                QuotationMarkStringMatch(text_segment2, 1, 2),
                QuotationMarkStringMatch(text_segment2, 18, 19),
                QuotationMarkStringMatch(text_segment2, 25, 26),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, text_segment1, 0, 1),
        QuotationMarkMetadata("\u201c", 2, QuotationMarkDirection.OPENING, text_segment1, 11, 12),
        QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.OPENING, text_segment2, 0, 1),
        QuotationMarkMetadata("\u201d", 2, QuotationMarkDirection.OPENING, text_segment2, 1, 2),
        QuotationMarkMetadata("\u201d", 2, QuotationMarkDirection.CLOSING, text_segment2, 18, 19),
        QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, text_segment2, 25, 26),
    ]
    assert western_european_quotation_mark_resolver.get_issues() == set()


def test_malformed_quotation_marks() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment1 = TextSegment.Builder().set_text("\u201c This is a,\u2018 quote").build()
    text_segment2 = (
        TextSegment.Builder()
        .set_text("This is the rest \u2019 of it \u201d")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment1, 0, 1),
                QuotationMarkStringMatch(text_segment1, 12, 13),
                QuotationMarkStringMatch(text_segment2, 17, 18),
                QuotationMarkStringMatch(text_segment2, 25, 26),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment1, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment1, 12, 13),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment2, 17, 18),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment2, 25, 26),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == set()


def test_unpaired_quotation_mark_issue() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = TextSegment.Builder().set_text("\u201cThis is a \u2018quote\u2019").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }

    text_segment = TextSegment.Builder().set_text("another quote\u201d").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 13, 14),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment, 13, 14),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK
    }


def test_too_deep_nesting_issue() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = (
        TextSegment.Builder().set_text("\u201cThis \u2018is \u201ca \u2018quote \u201cnested too deeply").build()
    )
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 6, 7),
                QuotationMarkStringMatch(text_segment, 10, 11),
                QuotationMarkStringMatch(text_segment, 13, 14),
                QuotationMarkStringMatch(text_segment, 20, 21),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, text_segment, 6, 7),
        QuotationMarkMetadata("\u201c", 3, QuotationMarkDirection.OPENING, text_segment, 10, 11),
        QuotationMarkMetadata("\u2018", 4, QuotationMarkDirection.OPENING, text_segment, 13, 14),
        # QuotationMarkMetadata("\u201c", 5, QuotationMarkDirection.Opening, text_segment, 20, 21),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.TOO_DEEP_NESTING,
        QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK,
    }


def test_incompatible_quotation_mark_issue() -> None:
    standard_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    )
    assert standard_english_quote_convention is not None
    standard_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_english_quote_convention])
    )
    standard_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_english_resolver_settings)

    text_segment = TextSegment.Builder().set_text("\u201cThis is a \u201cquote\u201d\u201d").build()
    assert list(
        standard_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u201c", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("\u201d", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert standard_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK
    }


def test_ambiguous_quotation_mark_issue() -> None:
    typewriter_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_english")
    )
    assert typewriter_english_quote_convention is not None
    typewriter_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([typewriter_english_quote_convention])
    )
    typewriter_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(typewriter_english_resolver_settings)

    text_segment = TextSegment.Builder().set_text('This"is an ambiguous quotation mark').build()
    assert (
        list(
            typewriter_english_quotation_mark_resolver.resolve_quotation_marks(
                [
                    QuotationMarkStringMatch(text_segment, 4, 5),
                ]
            )
        )
        == []
    )
    assert typewriter_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK
    }

    typewriter_english_quotation_mark_resolver.reset()
    text_segment = TextSegment.Builder().set_text("\u201cThis is an ambiguous quotation mark").build()
    assert (
        list(
            typewriter_english_quotation_mark_resolver.resolve_quotation_marks(
                [QuotationMarkStringMatch(text_segment, 0, 1)]
            )
        )
        == []
    )
    assert typewriter_english_quotation_mark_resolver.get_issues() == {
        QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK
    }


def test_typewriter_english_quotation_mark_recognition() -> None:
    typewriter_english_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_english")
    )
    assert typewriter_english_quote_convention is not None
    typewriter_english_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([typewriter_english_quote_convention])
    )
    typewriter_english_quotation_mark_resolver = DepthBasedQuotationMarkResolver(typewriter_english_resolver_settings)

    text_segment = (
        TextSegment.Builder().set_text("\"This is a 'quote'\"").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build()
    )
    assert list(
        typewriter_english_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata('"', 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("'", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("'", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata('"', 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert typewriter_english_quotation_mark_resolver.get_issues() == set()


def test_typewriter_french_mark_recognition() -> None:
    typewriter_french_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_french")
    )
    assert typewriter_french_quote_convention is not None
    typewriter_french_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([typewriter_french_quote_convention])
    )
    typewriter_french_quotation_mark_resolver = DepthBasedQuotationMarkResolver(typewriter_french_resolver_settings)

    text_segment = TextSegment.Builder().set_text("<<This is a <quote>>>").build()
    assert list(
        typewriter_french_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 2),
                QuotationMarkStringMatch(text_segment, 12, 13),
                QuotationMarkStringMatch(text_segment, 18, 19),
                QuotationMarkStringMatch(text_segment, 19, 21),
            ]
        )
    ) == [
        QuotationMarkMetadata("<<", 1, QuotationMarkDirection.OPENING, text_segment, 0, 2),
        QuotationMarkMetadata("<", 2, QuotationMarkDirection.OPENING, text_segment, 12, 13),
        QuotationMarkMetadata(">", 2, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
        QuotationMarkMetadata(">>", 1, QuotationMarkDirection.CLOSING, text_segment, 19, 21),
    ]
    assert typewriter_french_quotation_mark_resolver.get_issues() == set()


def test_central_european_quotation_mark_recognition() -> None:
    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None
    central_european_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([central_european_quote_convention])
    )
    central_european_quotation_mark_resolver = DepthBasedQuotationMarkResolver(central_european_resolver_settings)

    text_segment = (
        TextSegment.Builder()
        .set_text("\u201eThis is a \u201aquote\u2018\u201c")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        central_european_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201e", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u201a", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert central_european_quotation_mark_resolver.get_issues() == set()


def test_standard_swedish_quotation_mark_recognition() -> None:
    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    standard_swedish_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet([standard_swedish_quote_convention])
    )
    standard_swedish_quotation_mark_resolver = DepthBasedQuotationMarkResolver(standard_swedish_resolver_settings)

    text_segment = (
        TextSegment.Builder()
        .set_text("\u201dThis is a \u2019quote\u2019\u201d")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        standard_swedish_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert standard_swedish_quotation_mark_resolver.get_issues() == set()


def test_multiple_conventions_quotation_mark_recognition() -> None:
    typewriter_french_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_french")
    )
    assert typewriter_french_quote_convention is not None

    central_european_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    )
    assert central_european_quote_convention is not None

    standard_swedish_quote_convention = (
        standard_quote_conventions.STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_swedish")
    )
    assert standard_swedish_quote_convention is not None
    multiple_conventions_resolver_settings = QuoteConventionDetectionResolutionSettings(
        QuoteConventionSet(
            [typewriter_french_quote_convention, central_european_quote_convention, standard_swedish_quote_convention]
        )
    )
    multiple_conventions_quotation_mark_resolver = DepthBasedQuotationMarkResolver(
        multiple_conventions_resolver_settings
    )

    text_segment = (
        TextSegment.Builder()
        .set_text("\u201eThis is a \u2019quote>\u201c")
        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
        .build()
    )
    assert list(
        multiple_conventions_quotation_mark_resolver.resolve_quotation_marks(
            [
                QuotationMarkStringMatch(text_segment, 0, 1),
                QuotationMarkStringMatch(text_segment, 11, 12),
                QuotationMarkStringMatch(text_segment, 17, 18),
                QuotationMarkStringMatch(text_segment, 18, 19),
            ]
        )
    ) == [
        QuotationMarkMetadata("\u201e", 1, QuotationMarkDirection.OPENING, text_segment, 0, 1),
        QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.OPENING, text_segment, 11, 12),
        QuotationMarkMetadata(">", 2, QuotationMarkDirection.CLOSING, text_segment, 17, 18),
        QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.CLOSING, text_segment, 18, 19),
    ]
    assert multiple_conventions_quotation_mark_resolver.get_issues() == set()
