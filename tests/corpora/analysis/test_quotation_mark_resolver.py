from machine.corpora.analysis import QuotationMarkResolver, standard_quote_conventions


def test_reset() -> None:
    quotation_mark_resolver: QuotationMarkResolver = QuotationMarkResolver(
        standard_quote_conventions.standard_quote_conventions
    )

    assert quotation_mark_resolver._quotation_mark_resolver_state.quotation_stack == []
    assert quotation_mark_resolver._quotation_continuer_state.quotation_continuer_stack == []
    assert quotation_mark_resolver._quotation_mark_resolver_state.current_depth == 0
    assert quotation_mark_resolver._quotation_continuer_state.current_depth == 0

    quotation_mark_resolver.reset()

    assert quotation_mark_resolver._quotation_mark_resolver_state.quotation_stack == []
    assert quotation_mark_resolver._quotation_continuer_state.quotation_continuer_stack == []
    assert quotation_mark_resolver._quotation_mark_resolver_state.current_depth == 0
    assert quotation_mark_resolver._quotation_continuer_state.current_depth == 0
