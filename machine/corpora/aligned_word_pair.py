from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Iterator, List, Optional


@dataclass(unsafe_hash=True)
class AlignedWordPair:
    @classmethod
    def from_string(cls, alignments: str, invert: bool = False) -> Collection[AlignedWordPair]:
        result: List[AlignedWordPair] = []

        def convert_to_num(token: str) -> int:
            return -1 if token == "NULL" else int(token)

        for token in alignments.split():
            dash_index = token.index("-")
            i = convert_to_num(token[:dash_index])

            colon_index = token.find(":", dash_index + 1)
            if colon_index == -1:
                colon_index = len(token)
            j = convert_to_num(token[dash_index + 1 : colon_index])

            translation_score = -1
            alignment_score = -1

            second_colon_index = -1
            if colon_index < len(token):
                second_colon_index = token.find(":", colon_index + 1)
                if second_colon_index > 0:
                    translation_score = float(token[colon_index + 1 : second_colon_index])
                    alignment_score = float(token[second_colon_index + 1 : len(token)])
                else:
                    translation_score = float(token[colon_index + 1 : len(token)])

            result.append(
                AlignedWordPair(j, i, translation_score=translation_score, alignment_score=alignment_score)
                if invert
                else AlignedWordPair(i, j, translation_score=translation_score, alignment_score=alignment_score)
            )
        return result

    @classmethod
    def to_string(cls, word_pairs: Optional[Collection[AlignedWordPair]], include_scores: bool = True) -> str:
        if word_pairs is None:
            return ""
        return " ".join(wp._repr(include_scores) for wp in word_pairs)

    source_index: int
    target_index: int
    is_sure: bool = field(default=True, compare=False)
    translation_score: float = field(default=-1, compare=False)
    alignment_score: float = field(default=-1, compare=False)

    def invert(self) -> AlignedWordPair:
        return AlignedWordPair(
            self.target_index, self.source_index, self.is_sure, self.translation_score, self.alignment_score
        )

    def __iter__(self) -> Iterator[int]:
        return iter((self.source_index, self.target_index))

    def __repr__(self) -> str:
        return self._repr()

    def _repr(self, include_scores: bool = True) -> str:
        def format_score(score: float) -> str:
            return f"{score:.8f}".rstrip("0").rstrip(".")

        source_index = "NULL" if self.source_index < 0 else str(self.source_index)
        target_index = "NULL" if self.target_index < 0 else str(self.target_index)
        repr = f"{source_index}-{target_index}"
        if include_scores and self.translation_score >= 0:
            repr += f":{format_score(self.translation_score)}"
            if self.alignment_score >= 0:
                repr += f":{format_score(self.alignment_score)}"
        return repr
