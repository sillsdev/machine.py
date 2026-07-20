import math
from collections import defaultdict
from typing import Iterable, List, Type, cast, overload

from ..corpora import MultiKeyRef, ScriptureRef
from .score import Score
from .scripture_book_scores import ScriptureBookScores
from .scripture_book_usability import ScriptureBookUsability
from .scripture_chapter_scores import ScriptureChapterScores
from .scripture_chapter_usability import ScriptureChapterUsability
from .scripture_segment_score import ScriptureSegmentScore
from .scripture_segment_usability import ScriptureSegmentUsability
from .text_scores import TextScores
from .text_segment_score import TextSegmentScore
from .text_segment_usability import TextSegmentUsability
from .text_usability import TextUsability
from .thresholds import Thresholds
from .usability_parameters import UsabilityParameters


class ChrF3QualityEstimator:
    GREEN_THRESHOLD = 0.776
    YELLOW_THRESHOLD = 0.681

    def __init__(self, slope: float, intercept: float) -> None:
        self._slope = slope
        self._intercept = intercept
        self.book_thresholds = Thresholds(green_threshold=self.GREEN_THRESHOLD, yellow_threshold=self.YELLOW_THRESHOLD)
        self.chapter_thresholds = Thresholds(
            green_threshold=self.GREEN_THRESHOLD, yellow_threshold=self.YELLOW_THRESHOLD
        )
        self.segment_thresholds = Thresholds(
            green_threshold=self.GREEN_THRESHOLD, yellow_threshold=self.YELLOW_THRESHOLD
        )
        self.usable = UsabilityParameters.usable
        self.unusable = UsabilityParameters.unusable

    @overload
    def estimate_quality(
        self, confidences: Iterable[tuple[MultiKeyRef, float]], ref_type: Type[MultiKeyRef] = MultiKeyRef
    ) -> tuple[list[TextSegmentUsability], list[TextUsability]]: ...

    @overload
    def estimate_quality(
        self, confidences: Iterable[tuple[ScriptureRef, float]], ref_type: Type[ScriptureRef] = ScriptureRef
    ) -> tuple[list[ScriptureSegmentUsability], list[ScriptureChapterUsability], list[ScriptureBookUsability]]: ...

    def estimate_quality(
        self,
        confidences: Iterable[tuple[ScriptureRef, float] | tuple[MultiKeyRef, float]],
        ref_type: Type[ScriptureRef | MultiKeyRef] = ScriptureRef,
    ) -> (
        tuple[list[ScriptureSegmentUsability], list[ScriptureChapterUsability], list[ScriptureBookUsability]]
        | tuple[list[TextSegmentUsability], list[TextUsability]]
    ):
        confidences_list = list(confidences)
        if not confidences_list:
            return ([], []) if issubclass(ref_type, MultiKeyRef) else ([], [], [])

        first_key = confidences_list[0][0]
        if isinstance(first_key, ScriptureRef):
            segment_scores, chapter_scores, book_scores = self._project_chrf3_scripture(
                cast(List[tuple[ScriptureRef, float]], confidences_list)
            )
            return self._compute_segment_usability_scripture(segment_scores, chapter_scores, book_scores)
        else:
            segment_scores, text_scores = self._project_chrf3_text(
                cast(List[tuple[MultiKeyRef, float]], confidences_list)
            )
            return self._compute_segment_usability_text(segment_scores, text_scores)

    def _calculate_usable_probability(self, chrf3: float) -> float:
        usable_weight = math.exp(-((chrf3 - self.usable.mean) ** 2) / (2 * self.usable.variance)) * self.usable.count
        unusable_weight = (
            math.exp(-((chrf3 - self.unusable.mean) ** 2) / (2 * self.unusable.variance)) * self.unusable.count
        )
        return usable_weight / (usable_weight + unusable_weight)

    def _compute_book_usability(self, book_scores: ScriptureBookScores) -> list[ScriptureBookUsability]:
        usability_books = []
        for book in book_scores.scores.keys():
            score = book_scores.get_score(book)
            if score is None:
                continue
            book_usabilities = book_scores.get_segment_usabilities(book)
            average_probability = sum(book_usabilities) / len(book_usabilities) if book_usabilities else 0.0
            usability_books.append(
                ScriptureBookUsability(
                    book=book,
                    label=self.book_thresholds.return_label(average_probability),
                    usability=average_probability,
                    projected_chrf3=score.projected_chrf3,
                    confidence=score.confidence,
                )
            )
        return usability_books

    def _compute_chapter_usability(self, chapter_scores: ScriptureChapterScores) -> list[ScriptureChapterUsability]:
        usability_chapters = []
        for book, chapter_dict in chapter_scores.scores.items():
            for chapter in chapter_dict.keys():
                score = chapter_scores.get_score(book, chapter)
                if score is None:
                    continue
                chapter_usabilities = chapter_scores.get_segment_usabilities(book, chapter)
                average_probability = (
                    sum(chapter_usabilities) / len(chapter_usabilities) if chapter_usabilities else 0.0
                )
                usability_chapters.append(
                    ScriptureChapterUsability(
                        book=book,
                        chapter=chapter,
                        label=self.chapter_thresholds.return_label(average_probability),
                        usability=average_probability,
                        projected_chrf3=score.projected_chrf3,
                        confidence=score.confidence,
                    )
                )
        return usability_chapters

    def _compute_segment_usability_scripture(
        self,
        segment_scores: list[ScriptureSegmentScore],
        chapter_scores: ScriptureChapterScores,
        book_scores: ScriptureBookScores,
    ) -> tuple[list[ScriptureSegmentUsability], list[ScriptureChapterUsability], list[ScriptureBookUsability]]:
        usability_segments = []
        for segment_score in segment_scores:
            probability = self._calculate_usable_probability(segment_score.projected_chrf3)
            chapter_scores.append_segment_usability(
                segment_score.scripture_ref.book, segment_score.scripture_ref.chapter_num, probability
            )
            book_scores.append_segment_usability(segment_score.scripture_ref.book, probability)
            usability_segments.append(
                ScriptureSegmentUsability(
                    scripture_ref=segment_score.scripture_ref,
                    label=self.segment_thresholds.return_label(probability),
                    usability=probability,
                    projected_chrf3=segment_score.projected_chrf3,
                    confidence=segment_score.confidence,
                )
            )
        return (
            usability_segments,
            self._compute_chapter_usability(chapter_scores),
            self._compute_book_usability(book_scores),
        )

    def _compute_segment_usability_text(
        self, segment_scores: list[TextSegmentScore], text_scores: TextScores
    ) -> tuple[list[TextSegmentUsability], list[TextUsability]]:
        usability_segments = []
        for segment_score in segment_scores:
            probability = self._calculate_usable_probability(segment_score.projected_chrf3)
            text_scores.append_segment_usability(segment_score.text_id, probability)
            usability_segments.append(
                TextSegmentUsability(
                    segment_ref=segment_score.segment_ref,
                    label=self.segment_thresholds.return_label(probability),
                    usability=probability,
                    projected_chrf3=segment_score.projected_chrf3,
                    confidence=segment_score.confidence,
                )
            )
        return usability_segments, self._compute_text_usability(text_scores)

    def _compute_text_usability(self, text_scores: TextScores) -> list[TextUsability]:
        usability_texts = []
        for text_id in text_scores.scores.keys():
            score = text_scores.get_score(text_id)
            if score is None:
                continue
            text_usabilities = text_scores.get_segment_usabilities(text_id)
            average_probability = sum(text_usabilities) / len(text_usabilities) if text_usabilities else 0.0
            usability_texts.append(
                TextUsability(
                    text_id=text_id,
                    label=self.book_thresholds.return_label(average_probability),
                    usability=average_probability,
                    projected_chrf3=score.projected_chrf3,
                    confidence=score.confidence,
                )
            )
        return usability_texts

    @staticmethod
    def _gmean(values: List[float]) -> float:
        if not values or any(x <= 0 for x in values):
            return 0.0
        return math.exp(sum(math.log(x) for x in values) / len(values))

    def _project_chrf3_text(
        self, confidences: List[tuple[MultiKeyRef, float]]
    ) -> tuple[list[TextSegmentScore], TextScores]:
        confidences_by_text_id = defaultdict(list)
        segment_scores = []
        for segment_ref, confidence in confidences:
            score = TextSegmentScore(self._slope, confidence, self._intercept, segment_ref)
            segment_scores.append(score)
            confidences_by_text_id[segment_ref.text_id].append(confidence)

        text_scores = TextScores()
        for text_id, values in confidences_by_text_id.items():
            text_scores.add_score(text_id, Score(self._slope, self._gmean(values), self._intercept))
        return segment_scores, text_scores

    def _project_chrf3_scripture(
        self, confidences: List[tuple[ScriptureRef, float]]
    ) -> tuple[list[ScriptureSegmentScore], ScriptureChapterScores, ScriptureBookScores]:
        confidences_by_book = defaultdict(list)
        confidences_by_book_and_chapter = defaultdict(list)
        segment_scores = []
        for scripture_ref, confidence in confidences:
            score = ScriptureSegmentScore(self._slope, confidence, self._intercept, scripture_ref)
            segment_scores.append(score)
            book = scripture_ref.book
            chapter = scripture_ref.chapter_num
            confidences_by_book_and_chapter[(book, chapter)].append(confidence)
            confidences_by_book[book].append(confidence)

        chapter_scores = ScriptureChapterScores()
        for (book, chapter), values in confidences_by_book_and_chapter.items():
            chapter_scores.add_score(book, chapter, Score(self._slope, self._gmean(values), self._intercept))

        book_scores = ScriptureBookScores()
        for book, values in confidences_by_book.items():
            book_scores.add_score(book, Score(self._slope, self._gmean(values), self._intercept))
        return segment_scores, chapter_scores, book_scores
