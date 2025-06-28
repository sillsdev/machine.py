from .quotation_mark_direction import QuotationMarkDirection
from .quote_convention import QuoteConvention
from .text_segment import TextSegment


class QuotationMarkMetadata:
    def __init__(
        self,
        quotation_mark: str,
        depth: int,
        direction: QuotationMarkDirection,
        text_segment: TextSegment,
        start_index: int,
        end_index: int,
    ):
        self.quotation_mark = quotation_mark
        self.depth = depth
        self.direction = direction
        self.text_segment = text_segment
        self.start_index = start_index
        self.end_index = end_index

    def __eq__(self, other):
        if not isinstance(other, QuotationMarkMetadata):
            return False
        return (
            self.quotation_mark == other.quotation_mark
            and self.depth == other.depth
            and self.direction == other.direction
            and self.text_segment == other.text_segment
            and self.start_index == other.start_index
            and self.end_index == other.end_index
        )

    def get_quotation_mark(self) -> str:
        return self.quotation_mark

    def get_depth(self) -> int:
        return self.depth

    def get_direction(self) -> QuotationMarkDirection:
        return self.direction

    def get_text_segment(self) -> TextSegment:
        return self.text_segment

    def get_start_index(self) -> int:
        return self.start_index

    def get_end_index(self) -> int:
        return self.end_index

    def update_quotation_mark(self, quote_convention: QuoteConvention) -> None:
        updated_quotation_mark = quote_convention.get_expected_quotation_mark(self.depth, self.direction)
        if updated_quotation_mark == self.quotation_mark:
            return

        self.text_segment.replace_substring(
            self.start_index,
            self.end_index,
            updated_quotation_mark,
        )
