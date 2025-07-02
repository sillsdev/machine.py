from typing import List

from machine.corpora.analysis import (
    DepthBasedQuotationMarkResolver,
    QuotationMarkResolver,
    QuotationMarkStringMatch,
    QuoteConventionDetectionResolutionSettings,
    TextSegment,
    UsfmMarkerType,
    standard_quote_conventions,
)


def test_reset() -> None:
    quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
        QuoteConventionDetectionResolutionSettings(standard_quote_conventions.standard_quote_conventions)
    )

    assert quotation_mark_resolver._quotation_mark_resolver_state._quotation_stack == []
    assert quotation_mark_resolver._quotation_continuer_state._quotation_continuer_stack == []
    assert quotation_mark_resolver._quotation_mark_resolver_state._current_depth == 0
    assert quotation_mark_resolver._quotation_continuer_state._current_depth == 0

    quotation_mark_resolver.reset()

    assert quotation_mark_resolver._quotation_mark_resolver_state._quotation_stack == []
    assert quotation_mark_resolver._quotation_continuer_state._quotation_continuer_stack == []
    assert quotation_mark_resolver._quotation_mark_resolver_state._current_depth == 0
    assert quotation_mark_resolver._quotation_continuer_state._current_depth == 0

    quotation_mark_string_matches: List[QuotationMarkStringMatch] = [
        QuotationMarkStringMatch(TextSegment.Builder().set_text("Opening “quote").build(), 8, 9),
        QuotationMarkStringMatch(TextSegment.Builder().set_text("Another opening ‘quote").build(), 16, 17),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("“‘quote continuer").add_preceding_marker(UsfmMarkerType.PARAGRAPH).build(),
            0,
            1,
        ),
    ]

    list(quotation_mark_resolver.resolve_quotation_marks(quotation_mark_string_matches))
    assert len(quotation_mark_resolver._quotation_mark_resolver_state._quotation_stack) > 0
    assert quotation_mark_resolver._quotation_mark_resolver_state._current_depth > 0

    quotation_mark_resolver.reset()

    assert quotation_mark_resolver._quotation_mark_resolver_state._quotation_stack == []
    assert quotation_mark_resolver._quotation_continuer_state._quotation_continuer_stack == []
    assert quotation_mark_resolver._quotation_mark_resolver_state._current_depth == 0
    assert quotation_mark_resolver._quotation_continuer_state._current_depth == 0
