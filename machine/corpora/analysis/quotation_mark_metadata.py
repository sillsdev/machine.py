from dataclasses import dataclass

from .quotation_mark_direction import QuotationMarkDirection
from .quote_convention import QuoteConvention
from .text_segment import TextSegment


@dataclass
class QuotationMarkMetadata:

    quotation_mark: str
    depth: int
    direction: QuotationMarkDirection
    text_segment: TextSegment
    start_index: int
    end_index: int

    def update_quotation_mark(self, quote_convention: QuoteConvention) -> None:
        updated_quotation_mark = quote_convention.get_expected_quotation_mark(self.depth, self.direction)
        if updated_quotation_mark == self.quotation_mark:
            return

        self.text_segment.replace_substring(
            self.start_index,
            self.end_index,
            updated_quotation_mark,
        )
