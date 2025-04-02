from __future__ import annotations

from .scripture_update_block import ScriptureUpdateBlock


class ScriptureUpdateBlockHandlerBase:

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        raise NotImplementedError("Must be implemented in subclass")
