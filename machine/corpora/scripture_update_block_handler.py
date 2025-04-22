from __future__ import annotations

from abc import ABC, abstractmethod

from .scripture_update_block import ScriptureUpdateBlock


class ScriptureUpdateBlockHandler(ABC):

    @abstractmethod
    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock: ...
