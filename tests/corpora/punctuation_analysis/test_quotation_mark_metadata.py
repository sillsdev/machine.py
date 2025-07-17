from typing import Union

from machine.corpora.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuoteConvention,
    TextSegment,
)


def test_update_quotation_mark() -> None:
    quotation_mark_metadata = QuotationMarkMetadata(
        quotation_mark='"',
        depth=1,
        direction=QuotationMarkDirection.OPENING,
        text_segment=TextSegment.Builder().set_text('He said to the woman, "Has God really said,').build(),
        start_index=22,
        end_index=23,
    )
    quotation_mark_metadata.update_quotation_mark(get_quote_convention_by_name("standard_english"))
    assert quotation_mark_metadata.text_segment._text == "He said to the woman, “Has God really said,"

    quotation_mark_metadata = QuotationMarkMetadata(
        quotation_mark='"',
        depth=1,
        direction=QuotationMarkDirection.OPENING,
        text_segment=TextSegment.Builder().set_text('He said to the woman, "Has God really said,').build(),
        start_index=22,
        end_index=23,
    )
    quotation_mark_metadata.update_quotation_mark(get_quote_convention_by_name("western_european"))
    assert quotation_mark_metadata.text_segment._text == "He said to the woman, «Has God really said,"

    quotation_mark_metadata = QuotationMarkMetadata(
        quotation_mark='"',
        depth=1,
        direction=QuotationMarkDirection.OPENING,
        text_segment=TextSegment.Builder().set_text('He said to the woman, "Has God really said,').build(),
        start_index=23,
        end_index=24,
    )
    quotation_mark_metadata.update_quotation_mark(get_quote_convention_by_name("western_european"))
    assert quotation_mark_metadata.text_segment._text == 'He said to the woman, "«as God really said,'


def get_quote_convention_by_name(name: str) -> QuoteConvention:
    quote_convention: Union[QuoteConvention, None] = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(name)
    assert quote_convention is not None
    return quote_convention
