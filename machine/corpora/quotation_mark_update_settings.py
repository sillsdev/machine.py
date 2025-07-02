from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy


class QuotationMarkUpdateSettings:

    def __init__(
        self,
        default_chapter_action: QuotationMarkUpdateStrategy = QuotationMarkUpdateStrategy.APPLY_FULL,
        chapter_actions: list[QuotationMarkUpdateStrategy] = [],
    ):
        self._default_chapter_action = default_chapter_action
        self._chapter_actions = chapter_actions

    def get_action_for_chapter(self, chapter_number: int) -> QuotationMarkUpdateStrategy:
        if chapter_number <= len(self._chapter_actions):
            return self._chapter_actions[chapter_number - 1]
        return self._default_chapter_action
