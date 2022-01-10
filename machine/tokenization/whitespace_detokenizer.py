from typing import Any

from .string_detokenizer import DetokenizeOperation, StringDetokenizer


class WhitespaceDetokenizer(StringDetokenizer):
    def _get_operation(self, ctxt: Any, token: str) -> DetokenizeOperation:
        return DetokenizeOperation.NO_OPERATION
