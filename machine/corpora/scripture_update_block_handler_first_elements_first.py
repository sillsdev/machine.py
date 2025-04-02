from __future__ import annotations

from .scripture_update_block import ScriptureUpdateBlock
from .scripture_update_block_handler_base import ScriptureUpdateBlockHandlerBase
from .scripture_update_element import ScriptureUpdateElementType


class ScriptureUpdateBlockHandlerFirstElementsFirst(ScriptureUpdateBlockHandlerBase):

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        # If a paragraph, embed or style element occurs before existing text, move it before inserted text as well.
        current_insert_index = 0
        for current_index in range(len(block._elements)):
            element = block._elements[current_index]
            if element.type == ScriptureUpdateElementType.EXISTING_TEXT:
                # we found existing text, so we stop looking for elements to move
                break
            if current_index != current_insert_index and element.type != ScriptureUpdateElementType.INSERTED_TEXT:
                block._elements.remove(element)
                block._elements.insert(current_insert_index, element)
                current_insert_index += 1

        return block
