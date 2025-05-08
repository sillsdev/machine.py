from typing import Generator, Set, Union

from .analysis.quotation_mark_direction import QuotationMarkDirection
from .analysis.quotation_mark_metadata import QuotationMarkMetadata
from .analysis.quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .analysis.quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .analysis.quotation_mark_resolver import QuotationMarkResolver
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch


class BasicQuotationMarkResolver(QuotationMarkResolver):

    def __init__(self, settings: QuotationMarkResolutionSettings):
        self._settings: QuotationMarkResolutionSettings = settings
        self._last_quotation_mark: Union[QuotationMarkMetadata, None] = None
        self._issues: Set[QuotationMarkResolutionIssue] = set()

    def reset(self) -> None:
        self._last_quotation_mark = None
        self._issues = set()

    def resolve_quotation_marks(
        self, quote_matches: list[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for quote_match in quote_matches:
            yield from self._resolve_quotation_mark(quote_match)

    def _resolve_quotation_mark(
        self,
        quote_match: QuotationMarkStringMatch,
    ) -> Generator[QuotationMarkMetadata, None, None]:
        if self._is_opening_quote(quote_match):
            print("Opening quote: %s" % quote_match.get_context())
            quote: Union[QuotationMarkMetadata, None] = self._resolve_opening_mark(quote_match)
            if quote is not None:
                yield quote
            else:
                self._issues.add(QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK)
        elif self._is_closing_quote(quote_match):
            print("Closing quote: %s" % quote_match.get_context())
            quote: Union[QuotationMarkMetadata, None] = self._resolve_closing_mark(quote_match)
            if quote is not None:
                yield quote
            else:
                self._issues.add(QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK)
        else:
            print("Unknown quote %s" % quote_match.get_context())
            self._issues.add(QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK)

    def _is_opening_quote(
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
            or self._last_quotation_mark.get_direction() is not QuotationMarkDirection.Opening
        ):
            return False

        return (
            self._last_quotation_mark.get_text_segment() == match.get_text_segment()
            and self._last_quotation_mark.get_end_index() == match.get_start_index()
        )

    def _is_closing_quote(
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

    def _resolve_opening_mark(self, quote_match: QuotationMarkStringMatch) -> Union[QuotationMarkMetadata, None]:
        possible_depths: Set[int] = self._settings.get_possible_depths(
            quote_match.get_quotation_mark(), QuotationMarkDirection.Opening
        )
        if len(possible_depths) == 0:
            return None

        quote = quote_match.resolve(min(possible_depths), QuotationMarkDirection.Opening)
        self._last_quotation_mark = quote
        return quote

    def _resolve_closing_mark(self, quote_match: QuotationMarkStringMatch) -> Union[QuotationMarkMetadata, None]:
        possible_depths: Set[int] = self._settings.get_possible_depths(
            quote_match.get_quotation_mark(), QuotationMarkDirection.Closing
        )
        if len(possible_depths) == 0:
            return None

        quote = quote_match.resolve(min(possible_depths), QuotationMarkDirection.Closing)
        self._last_quotation_mark = quote
        return quote

    def get_issues(self) -> Set[QuotationMarkResolutionIssue]:
        return self._issues
