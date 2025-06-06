from .quotation_denormalization_action import QuotationDenormalizationAction


class QuotationDenormalizationSettings:

    def __init__(
        self,
        default_chapter_action: QuotationDenormalizationAction = QuotationDenormalizationAction.APPLY_FULL,
        chapter_actions: list[QuotationDenormalizationAction] = [],
    ):
        self._default_chapter_action = default_chapter_action
        self._chapter_actions = chapter_actions

    def get_action_for_chapter(self, chapter_number: int) -> QuotationDenormalizationAction:
        if chapter_number <= len(self._chapter_actions):
            return self._chapter_actions[chapter_number - 1]
        return self._default_chapter_action
