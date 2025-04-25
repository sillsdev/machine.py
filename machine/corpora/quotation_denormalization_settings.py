from .quotation_denormalization_action import QuotationDenormalizationAction


class QuotationDenormalizationSettings:

    def __init__(self):
        self._should_run_on_existing_text = False
        self._default_chapter_action = QuotationDenormalizationAction.APPLY_FULL
        self._chapter_actions: list[QuotationDenormalizationAction] = []

    def should_run_on_existing_text(self) -> bool:
        return self._should_run_on_existing_text

    def get_action_for_chapter(self, chapter_number: int) -> QuotationDenormalizationAction:
        if chapter_number <= len(self._chapter_actions):
            return self._chapter_actions[chapter_number - 1]
        return self._default_chapter_action

    class Builder:
        def __init__(self):
            self.settings = QuotationDenormalizationSettings()

        def run_on_existing_text(self) -> "QuotationDenormalizationSettings.Builder":
            self.settings._should_run_on_existing_text = True
            return self

        def set_chapter_actions(
            self, chapter_actions: list[QuotationDenormalizationAction]
        ) -> "QuotationDenormalizationSettings.Builder":
            self.settings._chapter_actions = chapter_actions
            return self

        def set_default_chapter_action(
            self, action: QuotationDenormalizationAction
        ) -> "QuotationDenormalizationSettings.Builder":
            self.settings._default_chapter_action = action
            return self

        def build(self):
            return self.settings
