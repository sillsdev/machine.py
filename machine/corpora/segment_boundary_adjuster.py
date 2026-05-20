from typing import List, Set, Tuple

import regex


# This class is used by SegmentBoundaryAdjuster when it is dealing with tokenized text.
class TokenRejoiner:

    _NO_TRAILING_SPACE_CHARACTERS: Set[str] = {"(", "[", "{", "«", "‹", "“", "‘"}
    _NO_LEADING_SPACE_CHARACTERS: Set[str] = {",", ";", ":", ".", "!", "?", ")", "]", "}", "”", "’", "»", "›"}

    def __init__(self) -> None:
        self._joined_text = ""
        self._num_tokens = 0

    @classmethod
    def join_tokens(cls, tokens: List[str]) -> str:
        rejoiner = cls()
        for token in tokens:
            rejoiner.add_token_to_joined_text(token)
        if len(rejoiner._joined_text) > 0 and rejoiner._joined_text[-1] not in cls._NO_TRAILING_SPACE_CHARACTERS:
            rejoiner._joined_text += " "
        return rejoiner._joined_text

    def add_token_to_joined_text(self, token: str) -> str:
        if self._num_tokens > 0:
            if (
                token not in self._NO_LEADING_SPACE_CHARACTERS
                and self._joined_text[-1] not in self._NO_TRAILING_SPACE_CHARACTERS
            ):
                self._joined_text += " "
        self._joined_text += token
        self._num_tokens += 1
        return self._joined_text


class SegmentBoundaryAdjuster:
    _PROHIBITED_VERSE_STARTING_CHARACTERS: Set[str] = {
        " ",
        ",",
        ";",
        ":",
        ".",
        "!",
        "?",
        ")",
        "]",
        "}",
        "”",
        "’",
    }
    _PROHIBITED_VERSE_ENDING_CHARACTERS: Set[str] = {"(", "[", "{", "«", "‹", "“", "‘"}
    _PUNCTUATION_AND_SENTENCE_STARTING_PATTERN = regex.compile(r".*([^\w\s]\s*)(\p{Lu}\w+(\s+\w+)?(\s+\w+)?\s*)$")
    _WORDS_AND_SENTENCE_ENDING_PATTERN = regex.compile(r"^(\p{Ll}\w+(\s+\w+)?(\s+\w+)?)([\.,;:!\?\)\]”’]\s*[”’]*\s*)")

    def adjust_segment_boundaries(self, verses: List[str]) -> List[str]:
        for i in range(len(verses) - 1):
            verses[i], verses[i + 1] = self.adjust_segment_pair_boundary(verses[i], verses[i + 1])
        return verses

    def adjust_segment_pair_boundary(self, segment: str, next_segment: str) -> Tuple[str, str]:
        while len(next_segment) > 0 and next_segment[0] in self._PROHIBITED_VERSE_STARTING_CHARACTERS:
            segment += next_segment[0]
            next_segment = next_segment[1:]
        while len(segment) > 0 and segment[-1] in self._PROHIBITED_VERSE_ENDING_CHARACTERS:
            next_segment = segment[-1] + next_segment
            segment = segment[:-1]
        if self._segment_ends_with_start_of_sentence(segment):
            segment, next_segment = self._adjust_for_missed_sentence_start(segment, next_segment)
        if self._segment_starts_with_end_of_sentence(next_segment):
            segment, next_segment = self._adjust_for_late_sentence_end(segment, next_segment)
        return segment, next_segment

    def _segment_ends_with_start_of_sentence(self, segment: str) -> bool:
        return self._PUNCTUATION_AND_SENTENCE_STARTING_PATTERN.match(segment) is not None

    def _adjust_for_missed_sentence_start(self, segment: str, next_segment: str) -> Tuple[str, str]:
        match = self._PUNCTUATION_AND_SENTENCE_STARTING_PATTERN.match(segment)
        if match is not None:
            capitalized_word = match.group(2)
            segment = segment[: match.end(1)]
            next_segment = capitalized_word + ("" if capitalized_word[-1] == " " else " ") + next_segment
        return segment, next_segment

    def _segment_starts_with_end_of_sentence(self, segment: str) -> bool:
        return self._WORDS_AND_SENTENCE_ENDING_PATTERN.match(segment) is not None

    def _adjust_for_late_sentence_end(self, segment: str, next_segment: str) -> Tuple[str, str]:
        match = self._WORDS_AND_SENTENCE_ENDING_PATTERN.match(next_segment)
        if match is not None:
            words = match.group(1)
            punctuation = match.group(4)
            segment = segment + words + punctuation
            next_segment = next_segment[match.end(0) :]
        return segment, next_segment

    def adjust_tokenized_segment_pair_boundaries(self, segment_boundary: int, tokens: List[str]) -> int:
        segment_text = TokenRejoiner.join_tokens(tokens[:segment_boundary])
        next_segment_text = TokenRejoiner.join_tokens(tokens[segment_boundary:])
        adjusted_segment_text = self.adjust_segment_pair_boundary(segment_text, next_segment_text)[0].strip()

        return self._find_best_boundary_from_segment_length(tokens, len(adjusted_segment_text))

    def _find_best_boundary_from_segment_length(self, tokens: List[str], target_segment_length: int) -> int:
        token_rejoiner = TokenRejoiner()

        for index, token in enumerate(tokens):
            accumulated_length = len(token_rejoiner.add_token_to_joined_text(token))

            if accumulated_length >= target_segment_length:
                # In the unlikely case that the adjusted boundary falls in the middle of a token
                # select the token boundary that is closest
                error_with_current_boundary = accumulated_length - target_segment_length
                error_with_previous_boundary = target_segment_length - (accumulated_length - len(token))

                if error_with_current_boundary < error_with_previous_boundary:
                    return index + 1
                else:
                    return index

        return len(tokens)
