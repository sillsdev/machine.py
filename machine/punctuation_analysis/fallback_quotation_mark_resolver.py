from typing import Generator, Optional, Set

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .quotation_mark_resolver import QuotationMarkResolver
from .quotation_mark_string_match import QuotationMarkStringMatch


class FallbackQuotationMarkResolver(QuotationMarkResolver):

    def __init__(self, settings: QuotationMarkResolutionSettings):
        self._settings: QuotationMarkResolutionSettings = settings
        self._last_quotation_mark: Optional[QuotationMarkMetadata] = None
        self._issues: Set[QuotationMarkResolutionIssue] = set()

    def reset(self) -> None:
        self._last_quotation_mark = None
        self._issues = set()

    def resolve_quotation_marks(
        self, quotation_mark_matches: list[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for quotation_mark_match in quotation_mark_matches:
            yield from self._resolve_quotation_mark(quotation_mark_match)

    def _resolve_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> Generator[QuotationMarkMetadata, None, None]:
        if self._is_opening_quotation_mark(quotation_mark_match):
            quotation_mark: Optional[QuotationMarkMetadata] = self._resolve_opening_mark(quotation_mark_match)
            if quotation_mark is not None:
                yield quotation_mark
            else:
                self._issues.add(QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK)
        elif self._is_closing_quotation_mark(quotation_mark_match):
            quotation_mark: Optional[QuotationMarkMetadata] = self._resolve_closing_mark(quotation_mark_match)
            if quotation_mark is not None:
                yield quotation_mark
            else:
                self._issues.add(QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK)
        else:
            # Make a reasonable guess about the direction of the quotation mark
            if (
                self._last_quotation_mark is None
                or self._last_quotation_mark.direction is QuotationMarkDirection.CLOSING
            ):
                quotation_mark: Optional[QuotationMarkMetadata] = self._resolve_opening_mark(quotation_mark_match)
                if quotation_mark is not None:
                    yield quotation_mark
            else:
                quotation_mark: Optional[QuotationMarkMetadata] = self._resolve_closing_mark(quotation_mark_match)
                if quotation_mark is not None:
                    yield quotation_mark

            self._issues.add(QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK)

    def _is_opening_quotation_mark(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:

        if self._settings.is_valid_opening_quotation_mark(match) and self._settings.is_valid_closing_quotation_mark(
            match
        ):
            return (
                match.is_at_start_of_segment()
                or match.has_leading_whitespace()
                or self._does_most_recent_opening_mark_immediately_precede(match)
                or match.has_quote_introducer_in_leading_substring()
            ) and not (match.has_trailing_whitespace() or match.has_trailing_punctuation())
        elif self._settings.is_valid_opening_quotation_mark(match):
            return True

        return False

    def _does_most_recent_opening_mark_immediately_precede(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:
        if (
            self._last_quotation_mark is None
            or self._last_quotation_mark.direction is not QuotationMarkDirection.OPENING
        ):
            return False

        return (
            self._last_quotation_mark.text_segment == match.text_segment
            and self._last_quotation_mark.end_index == match.start_index
        )

    def _is_closing_quotation_mark(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:

        if self._settings.is_valid_opening_quotation_mark(match) and self._settings.is_valid_closing_quotation_mark(
            match
        ):
            return (
                match.has_trailing_whitespace() or match.has_trailing_punctuation() or match.is_at_end_of_segment()
            ) and not match.has_leading_whitespace()
        elif self._settings.is_valid_closing_quotation_mark(match):
            return True

        return False

    def _resolve_opening_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> Optional[QuotationMarkMetadata]:
        possible_depths: Set[int] = self._settings.get_possible_depths(
            quotation_mark_match.quotation_mark, QuotationMarkDirection.OPENING
        )
        if len(possible_depths) == 0:
            return None

        quotation_mark = quotation_mark_match.resolve(min(possible_depths), QuotationMarkDirection.OPENING)
        self._last_quotation_mark = quotation_mark
        return quotation_mark

    def _resolve_closing_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> Optional[QuotationMarkMetadata]:
        possible_depths: Set[int] = self._settings.get_possible_depths(
            quotation_mark_match.quotation_mark, QuotationMarkDirection.CLOSING
        )
        if len(possible_depths) == 0:
            return None

        quotation_mark = quotation_mark_match.resolve(min(possible_depths), QuotationMarkDirection.CLOSING)
        self._last_quotation_mark = quotation_mark
        return quotation_mark

    def get_issues(self) -> Set[QuotationMarkResolutionIssue]:
        return self._issues
