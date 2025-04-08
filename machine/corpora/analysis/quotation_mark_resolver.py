from typing import Generator, Union

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention_set import QuoteConventionSet
from .usfm_marker_type import UsfmMarkerType


class QuotationMarkResolverState:

    def __init__(self):
        self.quotation_stack: list[QuotationMarkMetadata] = []
        self.current_depth: int = 0

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

    def get_deepest_opening_quotation_mark(self) -> str:
        if not self.has_open_quotation_mark():
            raise RuntimeError(
                "get_deepest_opening_quotation_mark() was called when the stack of quotation marks was empty."
            )
        return self.quotation_stack[-1].get_quotation_mark()


class QuotationContinuerState:
    def __init__(self):
        self.quotation_continuer_stack: list[QuotationMarkMetadata] = []

    def has_continuer_been_observed(self) -> bool:
        return len(self.quotation_continuer_stack) > 0

    def add_quotation_continuer(
        self, quote_match: QuotationMarkStringMatch, quotation_mark_resolver_state: QuotationMarkResolverState
    ) -> QuotationMarkMetadata:
        quote = quote_match.resolve(len(self.quotation_continuer_stack) + 1, QuotationMarkDirection.Opening)
        self.quotation_continuer_stack.append(quote)
        if len(self.quotation_continuer_stack) == len(quotation_mark_resolver_state.quotation_stack):
            self.quotation_continuer_stack.clear()
        return quote


class QuotationMarkResolver:
    quote_pattern = regex.compile(r"(?<=(.)|^)(\p{Quotation_Mark}|<<|>>|<|>)(?=(.)|$)", regex.U)
    apostrophe_pattern = regex.compile(r"[\'\u2019\u2018]", regex.U)
    whitespace_pattern = regex.compile(r"^[\s~]*$", regex.U)
    latin_letter_pattern = regex.compile(r"^\p{script=Latin}$", regex.U)
    punctuation_pattern = regex.compile(r"^[\.,;\?!\)\]\-—۔،؛]$", regex.U)

    def __init__(self, quote_convention_set: QuoteConventionSet):
        self.quote_convention_set = quote_convention_set
        self.quotation_mark_resolver_state = QuotationMarkResolverState()
        self.quotation_continuer_state = QuotationContinuerState()

    def resolve_quotation_marks(
        self, quote_matches: list[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for quote_index, quote_match in enumerate(quote_matches):
            previous_mark = None if quote_index == 0 else quote_matches[quote_index - 1]
            next_mark = None if quote_index == len(quote_matches) - 1 else quote_matches[quote_index + 1]
            yield from self._resolve_quotation_mark(quote_match, previous_mark, next_mark)

    def _resolve_quotation_mark(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_mark: Union[QuotationMarkStringMatch, None],
        next_mark: Union[QuotationMarkStringMatch, None],
    ) -> Generator[QuotationMarkMetadata, None, None]:
        if self._is_opening_quote(quote_match, previous_mark, next_mark):
            if self._is_quotation_continuer(quote_match, previous_mark, next_mark):
                quote = self._process_quotation_continuer(quote_match)
                yield quote
            else:
                if self._is_depth_too_great():
                    return

                quote = self._process_opening_mark(quote_match)
                yield quote
        elif self._is_apostrophe(quote_match, previous_mark, next_mark):
            pass
        elif self._is_closing_quote(quote_match, previous_mark, next_mark):
            if not self.quotation_mark_resolver_state.has_open_quotation_mark():
                return
            quote = self._process_closing_mark(quote_match)
            yield quote
        elif self._is_malformed_closing_quote(quote_match, previous_mark, next_mark):
            quote = self._process_closing_mark(quote_match)
            yield quote
        elif self._is_malformed_opening_quote(quote_match, previous_mark, next_mark):
            quote = self._process_opening_mark(quote_match)
            yield quote

    def _is_quotation_continuer(
        self,
        quote_match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if not quote_match.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.ParagraphMarker):
            return False
        if not self.quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        if not self.quotation_continuer_state.has_continuer_been_observed():
            if quote_match.start_index > 0:
                return False
            if (
                quote_match.get_quotation_mark()
                != self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark()
            ):
                return False
            if self.quotation_mark_resolver_state.are_more_than_n_quotes_open(1):
                if next_match is None or next_match.get_start_index() != quote_match.get_end_index():
                    return False
        else:
            if (
                quote_match.get_quotation_mark()
                != self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark()
            ):
                return False

        return True

    def _process_quotation_continuer(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        return self.quotation_continuer_state.add_quotation_continuer(quote_match, self.quotation_mark_resolver_state)

    def _is_depth_too_great(self) -> bool:
        return self.quotation_mark_resolver_state.are_more_than_n_quotes_open(4)

    def _process_opening_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        return self.quotation_mark_resolver_state.add_opening_quotation_mark(quote_match)

    def _process_closing_mark(self, quote_match: QuotationMarkStringMatch) -> QuotationMarkMetadata:
        return self.quotation_mark_resolver_state.add_closing_quotation_mark(quote_match)

    def _is_opening_quote(
        self,
        match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:

        if not match.is_valid_opening_quotation_mark(self.quote_convention_set):
            return False

        # if the quote convention is ambiguous, use whitespace as a clue
        if match.is_valid_closing_quotation_mark(self.quote_convention_set):
            return (
                match.has_leading_whitespace()
                or self._does_most_recent_opening_mark_immediately_precede(match)
                or match.has_leading_quote_introducer()
            ) and not (match.has_trailing_whitespace() or match.has_trailing_punctuation())
        return True

    def _is_closing_quote(
        self,
        match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:

        if not match.is_valid_closing_quotation_mark(self.quote_convention_set):
            return False

        # if the quote convention is ambiguous, use whitespace as a clue
        if self.quote_convention_set.is_valid_opening_quotation_mark(match.get_quotation_mark()):
            return (
                match.has_trailing_whitespace()
                or match.has_trailing_punctuation()
                or match.has_trailing_closing_quotation_mark(self.quote_convention_set)
            ) and not match.has_leading_whitespace()
        return True

    def _is_malformed_opening_quote(
        self,
        match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if not self.quote_convention_set.is_valid_opening_quotation_mark(match.get_quotation_mark()):
            return False

        if match.has_leading_quote_introducer():
            return True

        if (
            match.has_leading_whitespace()
            and match.has_trailing_whitespace()
            and not self.quotation_mark_resolver_state.has_open_quotation_mark()
        ):
            return True

        return False

    def _is_malformed_closing_quote(
        self,
        match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
        next_match: Union[QuotationMarkStringMatch, None],
    ) -> bool:
        if not self.quote_convention_set.is_valid_closing_quotation_mark(match.get_quotation_mark()):
            return False

        return (
            (
                not match.has_trailing_whitespace()
                or (match.has_leading_whitespace() and match.has_trailing_whitespace())
            )
            and self.quotation_mark_resolver_state.has_open_quotation_mark()
            and self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            )
        )

    def _does_most_recent_opening_mark_immediately_precede(self, match: QuotationMarkStringMatch) -> bool:
        if not self.quotation_mark_resolver_state.has_open_quotation_mark():
            return False

        return self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark() == match.get_previous_character()

    def _is_apostrophe(
        self,
        match: QuotationMarkStringMatch,
        previous_match: Union[QuotationMarkStringMatch, None],
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
            if not self.quotation_mark_resolver_state.has_open_quotation_mark():
                return True
            if not self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            ):
                return True
            if next_match is not None and self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), next_match.get_quotation_mark()
            ):
                return True

        # for languages that use apostrophes at the start and end of words
        if (
            not self.quotation_mark_resolver_state.has_open_quotation_mark()
            and match.get_quotation_mark() == "'"
            or self.quotation_mark_resolver_state.has_open_quotation_mark()
            and not self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_mark_resolver_state.get_deepest_opening_quotation_mark(), match.get_quotation_mark()
            )
        ):
            return True

        return False
