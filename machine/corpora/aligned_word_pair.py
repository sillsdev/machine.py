from dataclasses import dataclass, field
from typing import Set


@dataclass(unsafe_hash=True)
class AlignedWordPair:
    @classmethod
    def parse(cls, alignments: str, invert: bool = False) -> Set["AlignedWordPair"]:
        result: Set[AlignedWordPair] = set()
        for token in alignments.split():
            dash_index = token.index("-")
            i = int(token[:dash_index])

            colon_index = token.find(":", dash_index + 1)
            if colon_index == -1:
                colon_index = len(token)
            j = int(token[dash_index + 1 : colon_index])

            result.add(AlignedWordPair(j, i) if invert else AlignedWordPair(i, j))
        return result

    source_index: int
    target_index: int
    is_sure: bool = field(default=True, compare=False)
    translation_score: float = field(default=-1, compare=False)
    alignment_score: float = field(default=-1, compare=False)

    def invert(self) -> "AlignedWordPair":
        return AlignedWordPair(
            self.target_index, self.source_index, self.is_sure, self.translation_score, self.alignment_score
        )

    def __repr__(self) -> str:
        def format_score(score: float) -> str:
            return f"{score:.8f}".rstrip("0").rstrip(".")

        repr = f"{self.source_index}-{self.target_index}"
        if self.translation_score >= 0:
            repr += f":{format_score(self.translation_score)}"
            if self.alignment_score >= 0:
                repr += f":{format_score(self.alignment_score)}"
        return repr
