from dataclasses import dataclass

from ..utils.comparable import Comparable, compare


@dataclass(frozen=True)
class TextFileRef(Comparable):
    text_id: str
    line_num: int

    def compare_to(self, other: object) -> int:
        if not isinstance(other, TextFileRef):
            raise TypeError("other is not a TextFileRef object.")
        if self is other:
            return 0

        result = compare(self.text_id, other.text_id)
        if result != 0:
            return result

        return compare(self.line_num, other.line_num)

    def __hash__(self) -> int:
        code = 23
        code = code * 31 + hash(self.text_id)
        code = code * 31 + hash(self.line_num)
        return code

    def __repr__(self) -> str:
        return f"{self.text_id} {self.line_num}"
