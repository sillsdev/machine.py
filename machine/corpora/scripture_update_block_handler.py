from __future__ import annotations

from abc import ABC

from .scripture_update_block import ScriptureUpdateBlock


class ScriptureUpdateBlockHandler(ABC):

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock: ...
