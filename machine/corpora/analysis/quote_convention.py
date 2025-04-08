from .quotation_mark_direction import QuotationMarkDirection


class SingleLevelQuoteConvention:
    def __init__(self, opening_quote: str, closing_quote: str):
        self.opening_quote = opening_quote
        self.closing_quote = closing_quote

    def get_opening_quote(self) -> str:
        return self.opening_quote

    def get_closing_quote(self) -> str:
        return self.closing_quote


class QuoteConvention:
    def __init__(self, name: str, levels: list[SingleLevelQuoteConvention]):
        self.name = name
        self.levels = levels

    def get_name(self) -> str:
        return self.name

    def get_num_levels(self) -> int:
        return len(self.levels)

    def get_opening_quote_at_level(self, level: int) -> str:
        return self.levels[level - 1].get_opening_quote()

    def get_closing_quote_at_level(self, level: int) -> str:
        return self.levels[level - 1].get_closing_quote()

    def get_expected_quotation_mark(self, depth: int, direction: QuotationMarkDirection) -> str:
        if depth > len(self.levels):
            return ""
        return (
            self.get_opening_quote_at_level(depth)
            if direction == QuotationMarkDirection.Opening
            else self.get_closing_quote_at_level(depth)
        )

    def _includes_opening_quotation_mark(self, opening_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.get_opening_quote() == opening_quotation_mark:
                return True
        return False

    def _includes_closing_quotation_mark(self, closing_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.get_closing_quote() == closing_quotation_mark:
                return True
        return False

    def is_compatible_with_observed_quotation_marks(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> bool:
        for opening_quotation_mark in opening_quotation_marks:
            if not self._includes_opening_quotation_mark(opening_quotation_mark):
                return False
        for closing_quotation_mark in closing_quotation_marks:
            if not self._includes_closing_quotation_mark(closing_quotation_mark):
                return False

        # we require the first-level quotes to have been observed
        if self.get_opening_quote_at_level(1) not in opening_quotation_marks:
            return False
        if self.get_closing_quote_at_level(1) not in closing_quotation_marks:
            return False
        return True

    def print_summary(self) -> None:
        print(self.get_name())
        for level, convention in enumerate(self.levels):
            ordinal_name = self._get_ordinal_name(level + 1)
            print("%s%s-level quote%s" % (convention.get_opening_quote(), ordinal_name, convention.get_closing_quote()))

    def _get_ordinal_name(self, level) -> str:
        if level == 1:
            return "First"
        if level == 2:
            return "Second"
        if level == 3:
            return "Third"
        if level == 4:
            return "Fourth"
        return str(level) + "th"
