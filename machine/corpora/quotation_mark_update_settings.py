from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy


class QuotationMarkUpdateSettings:

    def __init__(
        self,
        default_chapter_strategy: QuotationMarkUpdateStrategy = QuotationMarkUpdateStrategy.APPLY_FULL,
        chapter_strategies: list[QuotationMarkUpdateStrategy] = [],
    ):
        self._default_chapter_strategy = default_chapter_strategy
        self._chapter_strategies = chapter_strategies

    def get_action_for_chapter(self, chapter_number: int) -> QuotationMarkUpdateStrategy:
        if chapter_number <= len(self._chapter_strategies):
            return self._chapter_strategies[chapter_number - 1]
        return self._default_chapter_strategy
