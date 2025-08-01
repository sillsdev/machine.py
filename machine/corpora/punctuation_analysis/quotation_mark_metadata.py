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

    @property
    def length(self) -> int:
        return self.end_index - self.start_index

    def shift_indices(self, shift_amount: int) -> None:
        self.start_index += shift_amount
        self.end_index += shift_amount

    def update_quotation_mark(self, quote_convention: QuoteConvention) -> None:
        updated_quotation_mark = quote_convention.get_expected_quotation_mark(self.depth, self.direction)
        if updated_quotation_mark == self.quotation_mark:
            return

        self.text_segment.replace_substring(
            self.start_index,
            self.end_index,
            updated_quotation_mark,
        )

        if len(updated_quotation_mark) != len(self.quotation_mark):
            self.end_index += len(updated_quotation_mark) - len(self.quotation_mark)
