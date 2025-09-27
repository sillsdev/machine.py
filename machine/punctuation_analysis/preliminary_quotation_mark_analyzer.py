from collections import Counter, defaultdict
from typing import Dict, Generator, List, Tuple

import regex

from .chapter import Chapter
from .quotation_mark_finder import QuotationMarkFinder
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention_set import QuoteConventionSet
from .text_segment import TextSegment
from .verse import Verse


class QuotationMarkCounter:
    _NEGLIGIBLE_PROPORTION_THRESHOLD = 0.01

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._quotation_mark_counts: Dict[str, int] = defaultdict(int)
        self._total_quotation_mark_count: int = 0

    def count_quotation_marks(self, quotation_marks: List[QuotationMarkStringMatch]) -> None:
        for quotation_mark_match in quotation_marks:
            mark: str = quotation_mark_match.quotation_mark
            self._quotation_mark_counts[mark] += 1
            self._total_quotation_mark_count += 1

    def is_quotation_mark_proportion_negligible(self, quotation_mark: str) -> bool:
        if self._total_quotation_mark_count == 0:
            return True
        return (
            self._quotation_mark_counts[quotation_mark] / self._total_quotation_mark_count
            < self._NEGLIGIBLE_PROPORTION_THRESHOLD
        )


class ApostropheProportionStatistics:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._num_characters = 0
        self._num_apostrophes = 0

    def count_characters(self, text_segment: TextSegment) -> None:
        self._num_characters += text_segment.length

    def add_apostrophe(self) -> None:
        self._num_apostrophes += 1

    def is_apostrophe_proportion_greater_than(self, threshold: float) -> bool:
        if self._num_characters == 0:
            return False
        return self._num_apostrophes / self._num_characters > threshold


class QuotationMarkWordPositions:
    _MAXIMUM_PROPORTION_FOR_RARITY = 0.1
    _MAXIMUM_PROPORTION_DIFFERENCE_THRESHOLD = 0.3

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._word_initial_occurrences: Counter[str] = Counter()
        self._mid_word_occurrences: Counter[str] = Counter()
        self._word_final_occurrences: Counter[str] = Counter()
        self._total_occurrences: Counter[str] = Counter()

    def count_word_initial_apostrophe(self, quotation_mark: str) -> None:
        self._word_initial_occurrences.update([quotation_mark])
        self._total_occurrences.update([quotation_mark])

    def count_mid_word_apostrophe(self, quotation_mark: str) -> None:
        self._mid_word_occurrences.update([quotation_mark])
        self._total_occurrences.update([quotation_mark])

    def count_word_final_apostrophe(self, quotation_mark: str) -> None:
        self._word_final_occurrences.update([quotation_mark])
        self._total_occurrences.update([quotation_mark])

    def _get_total_occurrences(self, quotation_mark: str) -> int:
        return (
            self._word_initial_occurrences[quotation_mark]
            + self._mid_word_occurrences[quotation_mark]
            + self._word_final_occurrences[quotation_mark]
        )

    def is_mark_rarely_initial(self, quotation_mark: str) -> bool:
        num_initial_marks: int = self._word_initial_occurrences[quotation_mark]
        num_total_marks: int = self._total_occurrences[quotation_mark]
        return num_total_marks > 0 and num_initial_marks / num_total_marks < self._MAXIMUM_PROPORTION_FOR_RARITY

    def is_mark_rarely_final(self, quotation_mark: str) -> bool:
        num_final_marks: int = self._word_final_occurrences[quotation_mark]
        num_total_marks: int = self._total_occurrences[quotation_mark]
        return num_total_marks > 0 and num_final_marks / num_total_marks < self._MAXIMUM_PROPORTION_FOR_RARITY

    def are_initial_and_final_rates_similar(self, quotation_mark: str) -> bool:
        num_initial_marks: int = self._word_initial_occurrences[quotation_mark]
        num_final_marks: int = self._word_final_occurrences[quotation_mark]
        num_total_marks: int = self._total_occurrences[quotation_mark]
        return (
            num_total_marks > 0
            and abs(num_initial_marks - num_final_marks) / num_total_marks
            < self._MAXIMUM_PROPORTION_DIFFERENCE_THRESHOLD
        )

    def is_mark_commonly_mid_word(self, quotation_mark: str) -> bool:
        num_mid_word_marks: int = self._mid_word_occurrences[quotation_mark]
        num_total_marks: int = self._total_occurrences[quotation_mark]
        return (
            num_total_marks > 0 and num_mid_word_marks / num_total_marks > self._MAXIMUM_PROPORTION_DIFFERENCE_THRESHOLD
        )


class QuotationMarkSequences:
    _SOLE_OCCURRENCE_MINIMUM_COUNT = 5
    _MUCH_MORE_COMMON_MINIMUM_RATIO = 10
    _MAXIMUM_PROPORTION_DIFFERENCE_THRESHOLD = 0.2

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._earlier_quotation_mark_counts: Counter[str] = Counter()
        self._later_quotation_mark_counts: Counter[str] = Counter()

    def count_earlier_quotation_mark(self, quotation_mark: str) -> None:
        self._earlier_quotation_mark_counts.update([quotation_mark])

    def count_later_quotation_mark(self, quotation_mark: str) -> None:
        self._later_quotation_mark_counts.update([quotation_mark])

    def is_mark_much_more_common_earlier(self, quotation_mark: str) -> bool:
        num_early_occurrences: int = self._earlier_quotation_mark_counts[quotation_mark]
        num_late_occurrences: int = self._later_quotation_mark_counts[quotation_mark]
        return (num_late_occurrences == 0 and num_early_occurrences > self._SOLE_OCCURRENCE_MINIMUM_COUNT) or (
            num_early_occurrences > num_late_occurrences * self._MUCH_MORE_COMMON_MINIMUM_RATIO
        )

    def is_mark_much_more_common_later(self, quotation_mark: str) -> bool:
        num_early_occurrences: int = self._earlier_quotation_mark_counts[quotation_mark]
        num_late_occurrences: int = self._later_quotation_mark_counts[quotation_mark]
        return (num_early_occurrences == 0 and num_late_occurrences > self._SOLE_OCCURRENCE_MINIMUM_COUNT) or (
            num_late_occurrences > num_early_occurrences * self._MUCH_MORE_COMMON_MINIMUM_RATIO
        )

    def are_early_and_late_mark_rates_similar(self, quotation_mark: str) -> bool:
        num_early_occurrences: int = self._earlier_quotation_mark_counts[quotation_mark]
        num_late_occurrences: int = self._later_quotation_mark_counts[quotation_mark]
        return (
            num_early_occurrences > 0
            and abs(num_late_occurrences - num_early_occurrences) / (num_early_occurrences + num_late_occurrences)
            < self._MAXIMUM_PROPORTION_DIFFERENCE_THRESHOLD
        )


class QuotationMarkGrouper:
    def __init__(self, quotation_marks: List[QuotationMarkStringMatch], quote_conventions: QuoteConventionSet):
        self._quote_conventions = quote_conventions
        self._group_quotation_marks(quotation_marks)

    def _group_quotation_marks(self, quotation_marks: List[QuotationMarkStringMatch]) -> None:
        self._grouped_quotation_marks: Dict[str, List[QuotationMarkStringMatch]] = defaultdict(list)
        for quotation_mark_match in quotation_marks:
            self._grouped_quotation_marks[quotation_mark_match.quotation_mark].append(quotation_mark_match)

    def get_quotation_mark_pairs(self) -> Generator[Tuple[str, str], None, None]:
        for mark1, matches1 in self._grouped_quotation_marks.items():
            # Handle cases of identical opening/closing marks
            if (
                len(matches1) == 2
                and self._quote_conventions.is_quotation_mark_direction_ambiguous(mark1)
                and not self.has_distinct_paired_quotation_mark(mark1)
            ):
                yield (mark1, mark1)
                continue

            # Skip verses where quotation mark pairs are ambiguous
            if len(matches1) > 1:
                continue

            # Find matching closing marks
            for mark2, matches2 in self._grouped_quotation_marks.items():
                if (
                    len(matches2) == 1
                    and self._quote_conventions.marks_are_a_valid_pair(mark1, mark2)
                    and matches1[0].precedes(matches2[0])
                ):
                    yield (mark1, mark2)

    def has_distinct_paired_quotation_mark(self, quotation_mark: str) -> bool:
        return any(
            [
                mark != quotation_mark and mark in self._grouped_quotation_marks
                for mark in self._quote_conventions.get_possible_paired_quotation_marks(quotation_mark)
            ]
        )


class PreliminaryApostropheAnalyzer:
    _APOSTROPHE_PATTERN = regex.compile(r"[\'\u2019]", regex.U)
    _MAXIMUM_APOSTROPHE_PROPORTION = 0.02

    def __init__(self):
        self._apostrophe_proportion_statistics = ApostropheProportionStatistics()
        self._word_position_statistics = QuotationMarkWordPositions()
        self.reset()

    def reset(self) -> None:
        self._apostrophe_proportion_statistics.reset()
        self._word_position_statistics.reset()

    def process_quotation_marks(
        self, text_segments: List[TextSegment], quotation_marks: List[QuotationMarkStringMatch]
    ) -> None:
        for text_segment in text_segments:
            self._apostrophe_proportion_statistics.count_characters(text_segment)
        for quotation_mark_match in quotation_marks:
            self._process_quotation_mark(quotation_mark_match)

    def _process_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> None:
        if quotation_mark_match.quotation_mark_matches(self._APOSTROPHE_PATTERN):
            self._count_apostrophe(quotation_mark_match)

    def _count_apostrophe(self, apostrophe_match: QuotationMarkStringMatch) -> None:
        apostrophe: str = apostrophe_match.quotation_mark
        self._apostrophe_proportion_statistics.add_apostrophe()
        if self._is_match_word_initial(apostrophe_match):
            self._word_position_statistics.count_word_initial_apostrophe(apostrophe)
        elif self._is_match_mid_word(apostrophe_match):
            self._word_position_statistics.count_mid_word_apostrophe(apostrophe)
        elif self._is_match_word_final(apostrophe_match):
            self._word_position_statistics.count_word_final_apostrophe(apostrophe)

    def _is_match_word_initial(self, apostrophe_match: QuotationMarkStringMatch) -> bool:
        if apostrophe_match.has_trailing_whitespace():
            return False
        if not apostrophe_match.is_at_start_of_segment() and not apostrophe_match.has_leading_whitespace():
            return False
        return True

    def _is_match_mid_word(self, apostrophe_match: QuotationMarkStringMatch) -> bool:
        if apostrophe_match.has_trailing_whitespace():
            return False
        if apostrophe_match.has_leading_whitespace():
            return False
        return True

    def _is_match_word_final(self, apostrophe_match: QuotationMarkStringMatch) -> bool:
        if not apostrophe_match.is_at_end_of_segment() and not apostrophe_match.has_trailing_whitespace():
            return False
        if apostrophe_match.has_leading_whitespace():
            return False
        return True

    def is_apostrophe_only(self, mark: str) -> bool:
        if not self._APOSTROPHE_PATTERN.search(mark):
            return False

        if self._word_position_statistics.is_mark_rarely_initial(
            mark
        ) or self._word_position_statistics.is_mark_rarely_final(mark):
            return True

        if self._word_position_statistics.are_initial_and_final_rates_similar(
            mark
        ) and self._word_position_statistics.is_mark_commonly_mid_word(mark):
            return True

        if self._apostrophe_proportion_statistics.is_apostrophe_proportion_greater_than(
            self._MAXIMUM_APOSTROPHE_PROPORTION
        ):
            return True

        return False


class PreliminaryQuotationMarkAnalyzer:

    def __init__(self, quote_conventions: QuoteConventionSet):
        self._quote_conventions = quote_conventions
        self._apostrophe_analyzer = PreliminaryApostropheAnalyzer()
        self._quotation_mark_sequences = QuotationMarkSequences()
        self._quotation_mark_counts = QuotationMarkCounter()
        self.reset()

    def reset(self) -> None:
        self._apostrophe_analyzer.reset()
        self._quotation_mark_sequences.reset()
        self._quotation_mark_counts.reset()

    def narrow_down_possible_quote_conventions(self, chapters: List[Chapter]) -> QuoteConventionSet:
        for chapter in chapters:
            self._analyze_quotation_marks_for_chapter(chapter)
        return self._select_compatible_quote_conventions()

    def _analyze_quotation_marks_for_chapter(self, chapter: Chapter) -> None:
        for verse in chapter.verses:
            self._analyze_quotation_marks_for_verse(verse)

    def _analyze_quotation_marks_for_verse(self, verse: Verse) -> None:
        quotation_marks: List[QuotationMarkStringMatch] = QuotationMarkFinder(
            self._quote_conventions
        ).find_all_potential_quotation_marks_in_verse(verse)
        self._analyze_quotation_mark_sequence(quotation_marks)
        self._apostrophe_analyzer.process_quotation_marks(verse.text_segments, quotation_marks)
        self._quotation_mark_counts.count_quotation_marks(quotation_marks)

    def _analyze_quotation_mark_sequence(self, quotation_marks: List[QuotationMarkStringMatch]) -> None:
        quotation_mark_grouper: QuotationMarkGrouper = QuotationMarkGrouper(quotation_marks, self._quote_conventions)
        for earlier_mark, later_mark in quotation_mark_grouper.get_quotation_mark_pairs():
            self._quotation_mark_sequences.count_earlier_quotation_mark(earlier_mark)
            self._quotation_mark_sequences.count_later_quotation_mark(later_mark)

    def _select_compatible_quote_conventions(self) -> QuoteConventionSet:
        opening_quotation_marks = self._find_opening_quotation_marks()
        closing_quotation_marks = self._find_closing_quotation_marks()

        return self._quote_conventions.filter_to_compatible_quote_conventions(
            opening_quotation_marks, closing_quotation_marks
        )

    def _find_opening_quotation_marks(self) -> List[str]:
        return [
            quotation_mark
            for quotation_mark in self._quote_conventions.get_possible_opening_marks()
            if self._is_opening_quotation_mark(quotation_mark)
        ]

    def _is_opening_quotation_mark(self, quotation_mark: str) -> bool:
        if self._quotation_mark_counts.is_quotation_mark_proportion_negligible(quotation_mark):
            return False
        if self._apostrophe_analyzer.is_apostrophe_only(quotation_mark):
            return False

        if self._quotation_mark_sequences.is_mark_much_more_common_earlier(quotation_mark):
            return True
        if self._quotation_mark_sequences.are_early_and_late_mark_rates_similar(
            quotation_mark
        ) and self._quote_conventions.is_quotation_mark_direction_ambiguous(quotation_mark):
            return True
        return False

    def _find_closing_quotation_marks(self) -> List[str]:
        return [
            quotation_mark
            for quotation_mark in self._quote_conventions.get_possible_closing_marks()
            if self._is_closing_quotation_mark(quotation_mark)
        ]

    def _is_closing_quotation_mark(self, quotation_mark: str) -> bool:
        if self._quotation_mark_counts.is_quotation_mark_proportion_negligible(quotation_mark):
            return False
        if self._apostrophe_analyzer.is_apostrophe_only(quotation_mark):
            return False

        if self._quotation_mark_sequences.is_mark_much_more_common_later(quotation_mark):
            return True

        if self._quotation_mark_sequences.are_early_and_late_mark_rates_similar(
            quotation_mark
        ) and self._quote_conventions.is_quotation_mark_direction_ambiguous(quotation_mark):
            return True
        return False
