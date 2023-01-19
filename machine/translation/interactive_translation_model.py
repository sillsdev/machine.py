from __future__ import annotations

from .interactive_translation_engine import InterativeTranslationEngine
from .translation_model import TranslationModel


class InteractiveTranslationModel(TranslationModel, InterativeTranslationEngine):
    def save(self) -> None:
        ...

    def __enter__(self) -> InteractiveTranslationModel:
        return self
