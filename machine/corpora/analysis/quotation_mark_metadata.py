from .quotation_mark_direction import QuotationMarkDirection


class QuotationMarkMetadata:
    def __init__(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection, start_index: int, end_index: int
    ):
        self.quotation_mark = quotation_mark
        self.depth = depth
        self.direction = direction
        self.start_index = start_index
        self.end_index = end_index

    def get_quotation_mark(self) -> str:
        return self.quotation_mark

    def get_depth(self) -> int:
        return self.depth

    def get_direction(self) -> QuotationMarkDirection:
        return self.direction

    def get_start_index(self) -> int:
        return self.start_index

    def get_end_index(self) -> int:
        return self.end_index
