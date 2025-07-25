from enum import Enum, auto
from typing import Generator, Optional, Set

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .quotation_mark_resolver import QuotationMarkResolver
from .quotation_mark_string_match import QuotationMarkStringMatch
from .usfm_marker_type import UsfmMarkerType


class QuotationMarkResolverState:

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._quotation_stack: list[QuotationMarkMetadata] = []

    @property
    def current_depth(self) -> int:
        return len(self._quotation_stack)

    def has_open_quotation_mark(self) -> bool:
        return self.current_depth > 0

    def are_more_than_n_quotes_open(self, n: int) -> bool:
        return self.current_depth > n

    def add_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        quotation_mark = quotation_mark_match.resolve(self.current_depth + 1, QuotationMarkDirection.OPENING)
        self._quotation_stack.append(quotation_mark)
        return quotation_mark

    def add_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        quotation_mark = quotation_mark_match.resolve(self.current_depth, QuotationMarkDirection.CLOSING)
        self._quotation_stack.pop()
        return quotation_mark

    def get_opening_quotation_mark_at_depth(self, depth: int) -> str:
        if depth > self.current_depth:
            raise RuntimeError(
                f"Opening quotation mark at depth ${depth} was requested from a quotation stack "
                + f"with depth ${self.current_depth}."
            )
        return self._quotation_stack[depth - 1].quotation_mark

    def get_deepest_opening_quotation_mark(self) -> str:
        if not self.has_open_quotation_mark():
            raise RuntimeError("The deepest opening quotation mark was requested from an empty quotation stack.")
        return self._quotation_stack[-1].quotation_mark


class QuoteContinuerStyle(Enum):
    UNDETERMINED = auto()
    ENGLISH = auto()
    SPANISH = auto()


class QuoteContinuerState:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._quote_continuer_mark_stack: list[QuotationMarkMetadata] = []
        self._continuer_style = QuoteContinuerStyle.UNDETERMINED

    @property
    def current_depth(self) -> int:
        return len(self._quote_continuer_mark_stack)

    def continuer_has_been_observed(self) -> bool:
        return len(self._quote_continuer_mark_stack) > 0

    @property
    def continuer_style(self) -> QuoteContinuerStyle:
        return self._continuer_style

    def add_quote_continuer(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
        quotation_mark_resolver_state: QuotationMarkResolverState,
        quote_continuer_style: QuoteContinuerStyle,
    ) -> QuotationMarkMetadata:
        quotation_mark = quotation_mark_match.resolve(
            len(self._quote_continuer_mark_stack) + 1, QuotationMarkDirection.OPENING
        )
        self._quote_continuer_mark_stack.append(quotation_mark)
        self._continuer_style = quote_continuer_style
        if self.current_depth == quotation_mark_resolver_state.current_depth:
            self._quote_continuer_mark_stack.clear()
        return quotation_mark


class QuotationMarkCategorizer:
    _APOSTROPHE_PATTERN = regex.compile(r"[\'\u2019\u2018]", regex.U)

    def __init__(
        self,
        quotation_mark_resolution_settings: QuotationMarkResolutionSettings,
        quotation_mark_resolver_state: QuotationMarkResolverState,
        quote_continuer_state: QuoteContinuerState,
    ):
        self._settings = quotation_mark_resolution_settings
        self._quotation_mark_resolver_state = quotation_mark_resolver_state
        self._quote_continuer_state = quote_continuer_state

    def is_english_quote_continuer(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
        previous_match: Optional[QuotationMarkStringMatch],
        next_match: Optional[QuotationMarkStringMatch],
    ) -> bool:
        if self._quote_continuer_state.continuer_style == QuoteContinuerStyle.SPANISH:
            return False
        if not self._meets_quote_continuer_prerequisites(quotation_mark_match):
            return False

        if (
            quotation_mark_match.quotation_mark
            != self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                self._quote_continuer_state.current_depth + 1
            )
        ):
            return False

        if not self._quote_continuer_state.continuer_has_been_observed():
            if quotation_mark_match._start_index > 0:
                return False

            # Check the next quotation mark match, since quote continuers must appear consecutively
            if self._quotation_mark_resolver_state.are_more_than_n_quotes_open(1):
                if next_match is None or next_match.start_index != quotation_mark_match.end_index:
                    return False

        return True

    def is_spanish_quote_continuer(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
        previous_match: Optional[QuotationMarkStringMatch],
        next_match: Optional[QuotationMarkStringMatch],
    ) -> bool:
        if self._quote_continuer_state.continuer_style == QuoteContinuerStyle.ENGLISH:
            return False
        if not self._meets_quote_continuer_prerequisites(quotation_mark_match):
            return False

        if not self._settings.are_marks_a_valid_pair(
            self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                self._quote_continuer_state.current_depth + 1
            ),
            quotation_mark_match.quotation_mark,
        ):
            return False

        if not self._quote_continuer_state.continuer_has_been_observed():
            if quotation_mark_match._start_index > 0:
                return False

            # This has only been observed with guillemets so far
            if quotation_mark_match.quotation_mark != "»":
                return False

            # Check the next quotation mark match, since quote continuers must appear consecutively
            if self._quotation_mark_resolver_state.are_more_than_n_quotes_open(1):
                if next_match is None or next_match.start_index != quotation_mark_match.end_index:
                    return False

        return True

    def _meets_quote_continuer_prerequisites(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:
        if self._quote_continuer_state.current_depth >= self._quotation_mark_resolver_state.current_depth:
            return False

        if (
            self._settings.should_rely_on_paragraph_markers
            and not quotation_mark_match._text_segment.marker_is_in_preceding_context(UsfmMarkerType.PARAGRAPH)
        ):
            return False
        if not self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return True

    def is_opening_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:

        if not self._settings.is_valid_opening_quotation_mark(quotation_mark_match):
            return False

        # If the quote convention is ambiguous, use whitespace as a clue
        if self._settings.is_valid_closing_quotation_mark(quotation_mark_match):
            return (
                quotation_mark_match.has_leading_whitespace()
                or self._most_recent_opening_mark_immediately_precedes(quotation_mark_match)
                or quotation_mark_match.has_quote_introducer_in_leading_substring()
            ) and not (
                quotation_mark_match.has_trailing_whitespace() or quotation_mark_match.has_trailing_punctuation()
            )
        return True

    def is_closing_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:

        if not self._settings.is_valid_closing_quotation_mark(quotation_mark_match):
            return False

        # If the quote convention is ambiguous, use whitespace as a clue
        if self._settings.is_valid_opening_quotation_mark(quotation_mark_match):
            return (
                quotation_mark_match.has_trailing_whitespace()
                or quotation_mark_match.has_trailing_punctuation()
                or quotation_mark_match.is_at_end_of_segment()
                or quotation_mark_match.next_character_matches(self._settings.closing_quotation_mark_regex)
            ) and not quotation_mark_match.has_leading_whitespace()
        return True

    def is_malformed_opening_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_opening_quotation_mark(quotation_mark_match):
            return False

        if quotation_mark_match.has_quote_introducer_in_leading_substring():
            return True

        if (
            quotation_mark_match.has_leading_whitespace()
            and quotation_mark_match.has_trailing_whitespace()
            and not self._quotation_mark_resolver_state.has_open_quotation_mark()
        ):
            return True

        return False

    def is_malformed_closing_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_closing_quotation_mark(quotation_mark_match):
            return False

        return (
            (
                quotation_mark_match.is_at_end_of_segment()
                or not quotation_mark_match.has_trailing_whitespace()
                or (quotation_mark_match.has_leading_whitespace() and quotation_mark_match.has_trailing_whitespace())
            )
            and self._quotation_mark_resolver_state.has_open_quotation_mark()
            and self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(),
                quotation_mark_match.quotation_mark,
            )
        )

    def is_unpaired_closing_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_closing_quotation_mark(quotation_mark_match):
            return False

        if self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return not quotation_mark_match.has_leading_whitespace() and (
            quotation_mark_match.is_at_end_of_segment() or quotation_mark_match.has_trailing_whitespace()
        )

    def _most_recent_opening_mark_immediately_precedes(self, match: QuotationMarkStringMatch) -> bool:
        if not self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == match.previous_character

    def is_apostrophe(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
        next_match: Optional[QuotationMarkStringMatch],
    ) -> bool:
        if not quotation_mark_match.quotation_mark_matches(self._APOSTROPHE_PATTERN):
            return False

        # Latin letters on both sides of punctuation mark
        if (
            quotation_mark_match.previous_character is not None
            and quotation_mark_match.has_leading_latin_letter()
            and quotation_mark_match.next_character is not None
            and quotation_mark_match.has_trailing_latin_letter()
        ):
            return True

        # Potential final s possessive (e.g. Moses')
        if quotation_mark_match.previous_character_matches(regex.compile(r"s")) and (
            quotation_mark_match.has_trailing_whitespace() or quotation_mark_match.has_trailing_punctuation()
        ):
            # Check whether it could be a closing quotation mark
            if not self._quotation_mark_resolver_state.has_open_quotation_mark():
                return True
            if not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(),
                quotation_mark_match.quotation_mark,
            ):
                return True
            if next_match is not None and self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(),
                next_match.quotation_mark,
            ):
                return True

        # For languages that use apostrophes at the start and end of words
        if (
            not self._quotation_mark_resolver_state.has_open_quotation_mark()
            and quotation_mark_match.quotation_mark == "'"
            or self._quotation_mark_resolver_state.has_open_quotation_mark()
            and not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(),
                quotation_mark_match.quotation_mark,
            )
        ):
            return True

        return False


class DepthBasedQuotationMarkResolver(QuotationMarkResolver):
    def __init__(self, settings: QuotationMarkResolutionSettings):
        self._settings = settings
        self._quotation_mark_resolver_state = QuotationMarkResolverState()
        self._quote_continuer_state = QuoteContinuerState()
        self._quotation_mark_categorizer = QuotationMarkCategorizer(
            self._settings, self._quotation_mark_resolver_state, self._quote_continuer_state
        )
        self._issues: Set[QuotationMarkResolutionIssue] = set()

    def reset(self) -> None:
        self._quotation_mark_resolver_state.reset()
        self._quote_continuer_state.reset()
        self._issues = set()

    def resolve_quotation_marks(
        self, quotation_mark_matches: list[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for index, quotation_mark_match in enumerate(quotation_mark_matches):
            previous_mark = None if index == 0 else quotation_mark_matches[index - 1]
            next_mark = None if index == len(quotation_mark_matches) - 1 else quotation_mark_matches[index + 1]
            yield from self._resolve_quotation_mark(quotation_mark_match, previous_mark, next_mark)
        if self._quotation_mark_resolver_state.has_open_quotation_mark():
            self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)

    def _resolve_quotation_mark(
        self,
        quotation_mark_match: QuotationMarkStringMatch,
        previous_mark: Optional[QuotationMarkStringMatch],
        next_mark: Optional[QuotationMarkStringMatch],
    ) -> Generator[QuotationMarkMetadata, None, None]:
        if self._quotation_mark_categorizer.is_opening_quotation_mark(quotation_mark_match):
            if self._quotation_mark_categorizer.is_english_quote_continuer(
                quotation_mark_match, previous_mark, next_mark
            ):
                yield self._process_quote_continuer(quotation_mark_match, QuoteContinuerStyle.ENGLISH)
            else:
                if self._is_depth_too_great():
                    self._issues.add(QuotationMarkResolutionIssue.TOO_DEEP_NESTING)
                    return

                yield self._process_opening_mark(quotation_mark_match)
        elif self._quotation_mark_categorizer.is_apostrophe(quotation_mark_match, next_mark):
            pass
        elif self._quotation_mark_categorizer.is_closing_quotation_mark(quotation_mark_match):
            if self._quotation_mark_categorizer.is_spanish_quote_continuer(
                quotation_mark_match, previous_mark, next_mark
            ):
                yield self._process_quote_continuer(quotation_mark_match, QuoteContinuerStyle.SPANISH)
            elif not self._quotation_mark_resolver_state.has_open_quotation_mark():
                self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)
                return
            else:
                yield self._process_closing_mark(quotation_mark_match)
        elif self._quotation_mark_categorizer.is_malformed_closing_quotation_mark(quotation_mark_match):
            yield self._process_closing_mark(quotation_mark_match)
        elif self._quotation_mark_categorizer.is_malformed_opening_quotation_mark(quotation_mark_match):
            yield self._process_opening_mark(quotation_mark_match)
        elif self._quotation_mark_categorizer.is_unpaired_closing_quotation_mark(quotation_mark_match):
            self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)
        else:
            self._issues.add(QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK)

    def _process_quote_continuer(
        self, quotation_mark_match: QuotationMarkStringMatch, continuer_style: QuoteContinuerStyle
    ) -> QuotationMarkMetadata:
        return self._quote_continuer_state.add_quote_continuer(
            quotation_mark_match, self._quotation_mark_resolver_state, continuer_style
        )

    def _is_depth_too_great(self) -> bool:
        return self._quotation_mark_resolver_state.are_more_than_n_quotes_open(3)

    def _process_opening_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        if not self._settings.metadata_matches_quotation_mark(
            quotation_mark_match.quotation_mark,
            self._quotation_mark_resolver_state.current_depth + 1,
            QuotationMarkDirection.OPENING,
        ):
            self._issues.add(QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK)
        return self._quotation_mark_resolver_state.add_opening_quotation_mark(quotation_mark_match)

    def _process_closing_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        if not self._settings.metadata_matches_quotation_mark(
            quotation_mark_match.quotation_mark,
            self._quotation_mark_resolver_state.current_depth,
            QuotationMarkDirection.CLOSING,
        ):
            self._issues.add(QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK)
        return self._quotation_mark_resolver_state.add_closing_quotation_mark(quotation_mark_match)

    def get_issues(self) -> Set[QuotationMarkResolutionIssue]:
        return self._issues
