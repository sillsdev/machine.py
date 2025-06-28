from enum import Enum, auto
from typing import Generator, Set, Union

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
        self.quotation_stack: list[QuotationMarkMetadata] = []
        self.current_depth: int = 0

    def get_current_depth(self) -> int:
        return self.current_depth + 1

    def has_open_quotation_mark(self) -> bool:
        return self.current_depth > 0

    def are_more_than_n_quotes_open(self, n: int) -> bool:
        return self.current_depth > n

    def add_opening_quotation_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        quote = quote_match.resolve(self.current_depth + 1, QuotationMarkDirection.Opening)
        self.quotation_stack.append(quote)
        self.current_depth += 1
        return quote

    def add_closing_quotation_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        quote = quote_match.resolve(self.current_depth, QuotationMarkDirection.Closing)
        self.quotation_stack.pop()
        self.current_depth -= 1
        return quote

    def get_opening_quotation_mark_at_depth(self, depth: int) -> str:
        if depth > len(self.quotation_stack):
            raise RuntimeError(
                "get_opening_quotation_mark_at_depth() was called with a depth greater than the quotation stack size."
            )
        return self.quotation_stack[depth - 1].get_quotation_mark()

    def get_deepest_opening_quotation_mark(self) -> str:
        if not self.has_open_quotation_mark():
            raise RuntimeError(
                "get_deepest_opening_quotation_mark() was called when the stack of quotation marks was empty."
            )
        return self.quotation_stack[-1].get_quotation_mark()


class QuotationContinuerStyle(Enum):
    UNDETERMINED = auto()
    ENGLISH = auto()
    SPANISH = auto()


class QuotationContinuerState:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.quotation_continuer_stack: list[QuotationMarkMetadata] = []
        self.current_depth = 0
        self.continuer_style = QuotationContinuerStyle.UNDETERMINED

    def get_current_depth(self) -> int:
        return self.current_depth

    def has_continuer_been_observed(self) -> bool:
        return len(self.quotation_continuer_stack) > 0

    def get_continuer_style(self) -> QuotationContinuerStyle:
        return self.continuer_style

    def add_quotation_continuer(
        self,
        quote_match: QuotationMarkStringMatch,
        quotation_mark_resolver_state: QuotationMarkResolverState,
        quotation_continuer_style: QuotationContinuerStyle,
    ) -> QuotationMarkMetadata:
        quote = quote_match.resolve(len(self.quotation_continuer_stack) + 1, QuotationMarkDirection.Opening)
        self.quotation_continuer_stack.append(quote)
        self.current_depth += 1
        self.continuer_style = quotation_continuer_style
        if len(self.quotation_continuer_stack) == len(quotation_mark_resolver_state.quotation_stack):
            self.quotation_continuer_stack.clear()
            self.current_depth = 0
        return quote


class QuotationMarkCategorizer:
    apostrophe_pattern = regex.compile(r"[\'\u2019\u2018]", regex.U)

    def __init__(
        self,
        quotation_mark_resolution_settings: QuotationMarkResolutionSettings,
        quotation_mark_resolver_state: QuotationMarkResolverState,
        quotation_continuer_state: QuotationContinuerState,
    ):
        self._settings = quotation_mark_resolution_settings
        self._quotation_mark_resolver_state = quotation_mark_resolver_state
        self._quotation_continuer_state = quotation_continuer_state

    def is_english_quotation_continuer(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if self._quotation_continuer_state.get_continuer_style() == QuotationContinuerStyle.SPANISH:
            return False
        if not self._meets_quote_continuer_prerequisites(quote_match, previous_match, next_match):
            return False

        if not self._quotation_continuer_state.has_continuer_been_observed():
            if quote_match.start_index > 0:
                return False
            if (
                quote_match.get_quotation_mark()
                != self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                    self._quotation_continuer_state.get_current_depth() + 1
                )
            ):
                return False
            if self._quotation_mark_resolver_state.are_more_than_n_quotes_open(1):
                if next_match is None or next_match.get_start_index() != quote_match.get_end_index():
                    return False
        else:
            if (
                quote_match.get_quotation_mark()
                != self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                    self._quotation_continuer_state.get_current_depth() + 1
                )
            ):
                return False

        return True

    def is_spanish_quotation_continuer(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if self._quotation_continuer_state.get_continuer_style() == QuotationContinuerStyle.ENGLISH:
            return False
        if not self._meets_quote_continuer_prerequisites(quote_match, previous_match, next_match):
            return False

        if not self._quotation_continuer_state.has_continuer_been_observed():
            if quote_match.start_index > 0:
                return False

            # this has only been observed with guillemets so far
            if quote_match.get_quotation_mark() != "»":
                return False
            if not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                    self._quotation_continuer_state.get_current_depth() + 1
                ),
                quote_match.get_quotation_mark(),
            ):
                return False
            if self._quotation_mark_resolver_state.are_more_than_n_quotes_open(1):
                if next_match is None or next_match.get_start_index() != quote_match.get_end_index():
                    return False
        else:
            if not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_opening_quotation_mark_at_depth(
                    self._quotation_continuer_state.get_current_depth() + 1
                ),
                quote_match.get_quotation_mark(),
            ):
                return False

        return True

    def _meets_quote_continuer_prerequisites(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if (
            self._settings.should_rely_on_paragraph_markers()
            and not quote_match.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.ParagraphMarker)
        ):
            return False
        if not self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return True

    def is_opening_quote(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:

        if not self._settings.is_valid_opening_quotation_mark(match):
            return False

        # if the quote convention is ambiguous, use whitespace as a clue
        if self._settings.is_valid_closing_quotation_mark(match):
            return (
                match.has_leading_whitespace()
                or self._does_most_recent_opening_mark_immediately_precede(match)
                or match.has_quote_introducer_in_leading_substring()
            ) and not (match.has_trailing_whitespace() or match.has_trailing_punctuation())
        return True

    def is_closing_quote(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:

        if not self._settings.is_valid_closing_quotation_mark(match):
            return False

        # if the quote convention is ambiguous, use whitespace as a clue
        if self._settings.is_valid_opening_quotation_mark(match):
            return (
                match.has_trailing_whitespace()
                or match.has_trailing_punctuation()
                or match.is_at_end_of_segment()
                or match.does_next_character_match(self._settings.get_closing_quotation_mark_regex())
            ) and not match.has_leading_whitespace()
        return True

    def is_malformed_opening_quote(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_opening_quotation_mark(match):
            return False

        if match.has_quote_introducer_in_leading_substring():
            return True

        if (
            match.has_leading_whitespace()
            and match.has_trailing_whitespace()
            and not self._quotation_mark_resolver_state.has_open_quotation_mark()
        ):
            return True

        return False

    def is_malformed_closing_quote(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_closing_quotation_mark(match):
            return False

        return (
            (
                match.is_at_end_of_segment()
                or not match.has_trailing_whitespace()
                or (match.has_leading_whitespace() and match.has_trailing_whitespace())
            )
            and self._quotation_mark_resolver_state.has_open_quotation_mark()
            and self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            )
        )

    def is_unpaired_closing_quote(
        self,
        match: QuotationMarkStringMatch,
    ) -> bool:
        if not self._settings.is_valid_closing_quotation_mark(match):
            return False

        if self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return not match.has_leading_whitespace() and (match.is_at_end_of_segment() or match.has_trailing_whitespace())

    def _does_most_recent_opening_mark_immediately_precede(self, match: QuotationMarkStringMatch) -> bool:
        if not self._quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return (
            self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == match.get_previous_character()
        )

    def is_apostrophe(
        self,
        match: QuotationMarkStringMatch,
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if not match.does_quotation_mark_match(self.apostrophe_pattern):
            return False

        # Latin letters on both sides of punctuation mark
        if (
            match.get_previous_character() is not None
            and match.has_leading_latin_letter()
            and match.get_next_character() is not None
            and match.has_trailing_latin_letter()
        ):
            return True

        # potential final s possessive (e.g. Moses')
        if match.does_previous_character_match(regex.compile(r"s")) and (
            match.has_trailing_whitespace() or match.has_trailing_punctuation()
        ):
            # check whether it could be a closing quote
            if not self._quotation_mark_resolver_state.has_open_quotation_mark():
                return True
            if not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            ):
                return True
            if next_match is not None and self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(),
                next_match.get_quotation_mark(),
            ):
                return True

        # for languages that use apostrophes at the start and end of words
        if (
            not self._quotation_mark_resolver_state.has_open_quotation_mark()
            and match.get_quotation_mark() == "'"
            or self._quotation_mark_resolver_state.has_open_quotation_mark()
            and not self._settings.are_marks_a_valid_pair(
                self._quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            )
        ):
            return True

        return False


class DepthBasedQuotationMarkResolver(QuotationMarkResolver):
    def __init__(self, settings: QuotationMarkResolutionSettings):
        self._settings = settings
        self._quotation_mark_resolver_state = QuotationMarkResolverState()
        self._quotation_continuer_state = QuotationContinuerState()
        self._quotation_mark_categorizer = QuotationMarkCategorizer(
            self._settings, self._quotation_mark_resolver_state, self._quotation_continuer_state
        )
        self._issues: Set[QuotationMarkResolutionIssue] = set()

    def reset(self) -> None:
        self._quotation_mark_resolver_state.reset()
        self._quotation_continuer_state.reset()
        self._issues = set()

    def resolve_quotation_marks(
        self, quote_matches: list[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for quote_index, quote_match in enumerate(quote_matches):
            previous_mark = None if quote_index == 0 else quote_matches[quote_index - 1]
            next_mark = None if quote_index == len(quote_matches) - 1 else quote_matches[quote_index + 1]
            yield from self._resolve_quotation_mark(quote_match, previous_mark, next_mark)
        if self._quotation_mark_resolver_state.has_open_quotation_mark():
            self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)

    def _resolve_quotation_mark(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_mark: Union[QuotationMarkStringMatch, None],
        next_mark: Union[QuotationMarkStringMatch, None],
    ) -> Generator[QuotationMarkMetadata, None, None]:
        if self._quotation_mark_categorizer.is_opening_quote(quote_match):
            if self._quotation_mark_categorizer.is_english_quotation_continuer(quote_match, previous_mark, next_mark):
                yield self._process_quotation_continuer(quote_match, QuotationContinuerStyle.ENGLISH)
            else:
                if self._is_depth_too_great():
                    self._issues.add(QuotationMarkResolutionIssue.TOO_DEEP_NESTING)
                    return

                yield self._process_opening_mark(quote_match)
        elif self._quotation_mark_categorizer.is_apostrophe(quote_match, next_mark):
            pass
        elif self._quotation_mark_categorizer.is_closing_quote(quote_match):
            if self._quotation_mark_categorizer.is_spanish_quotation_continuer(quote_match, previous_mark, next_mark):
                yield self._process_quotation_continuer(quote_match, QuotationContinuerStyle.SPANISH)
            elif not self._quotation_mark_resolver_state.has_open_quotation_mark():
                self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)
                return
            else:
                yield self._process_closing_mark(quote_match)
        elif self._quotation_mark_categorizer.is_malformed_closing_quote(quote_match):
            yield self._process_closing_mark(quote_match)
        elif self._quotation_mark_categorizer.is_malformed_opening_quote(quote_match):
            yield self._process_opening_mark(quote_match)
        elif self._quotation_mark_categorizer.is_unpaired_closing_quote(quote_match):
            self._issues.add(QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK)
        else:
            self._issues.add(QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK)

    def _process_quotation_continuer(
        self, quote_match: QuotationMarkStringMatch, continuer_style: QuotationContinuerStyle
    ) -> QuotationMarkMetadata:
        return self._quotation_continuer_state.add_quotation_continuer(
            quote_match, self._quotation_mark_resolver_state, continuer_style
        )

    def _is_depth_too_great(self) -> bool:
        return self._quotation_mark_resolver_state.are_more_than_n_quotes_open(3)

    def _process_opening_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        if not self._settings.does_metadata_match_quotation_mark(
            quote_match.get_quotation_mark(),
            self._quotation_mark_resolver_state.get_current_depth(),
            QuotationMarkDirection.Opening,
        ):
            self._issues.add(QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK)
        return self._quotation_mark_resolver_state.add_opening_quotation_mark(quote_match)

    def _process_closing_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        if not self._settings.does_metadata_match_quotation_mark(
            quote_match.get_quotation_mark(),
            self._quotation_mark_resolver_state.get_current_depth() - 1,
            QuotationMarkDirection.Closing,
        ):
            self._issues.add(QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK)
        return self._quotation_mark_resolver_state.add_closing_quotation_mark(quote_match)

    def get_issues(self) -> Set[QuotationMarkResolutionIssue]:
        return self._issues
