from __future__ import annotations

from abc import abstractmethod
from typing import ContextManager

from .translation_engine import TranslationEngine


class NmtTranslationEngine(TranslationEngine, ContextManager["NmtTranslationEngine"]):
    @abstractmethod
    def get_batch_size(self) -> int:
        ...
