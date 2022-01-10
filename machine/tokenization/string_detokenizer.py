from abc import abstractmethod
from enum import Enum, auto
from typing import Any, Iterable, List

from .detokenizer import Detokenizer


class DetokenizeOperation(Enum):
    NO_OPERATION = auto()
    MERGE_LEFT = auto()
    MERGE_RIGHT = auto()
    MERGE_BOTH = auto()


class StringDetokenizer(Detokenizer[str, str]):
    def detokenize(self, tokens: Iterable[str]) -> str:
        token_list = list(tokens)
        ctxt = self._create_context()
        ops = [self._get_operation(ctxt, t) for t in token_list]
        result = ""
        for i in range(len(token_list)):
            result += token_list[i]

            append_separator = True
            if i + 1 == len(ops):
                append_separator = False
            elif ops[i + 1] in {DetokenizeOperation.MERGE_LEFT, DetokenizeOperation.MERGE_BOTH}:
                append_separator = False
            elif ops[i] in {DetokenizeOperation.MERGE_RIGHT, DetokenizeOperation.MERGE_BOTH}:
                append_separator = False

            if append_separator:
                result += self._get_separator(token_list, ops, i)
        return result

    def _create_context(self) -> Any:
        return None

    @abstractmethod
    def _get_operation(self, ctxt: Any, token: str) -> DetokenizeOperation:
        ...

    def _get_separator(self, tokens: List[str], ops: List[DetokenizeOperation], index: int) -> str:
        return " "
