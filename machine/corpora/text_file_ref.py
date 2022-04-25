from dataclasses import dataclass

from ..utils.comparable import Comparable, compare


@dataclass(frozen=True)
class TextFileRef(Comparable):
    file_id: str
    line_num: int

    def compare_to(self, other: object) -> int:
        if not isinstance(other, TextFileRef):
            raise TypeError("other is not a TextFileRef object.")
        if self is other:
            return 0

        result = compare(self.file_id, other.file_id)
        if result != 0:
            return result

        return compare(self.line_num, other.line_num)

    def __hash__(self) -> int:
        code = 23
        code = code * 31 + hash(self.file_id)
        code = code * 31 + hash(self.line_num)
        return code

    def __repr__(self) -> str:
        return f"{self.file_id} {self.line_num}"
