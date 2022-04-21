from typing import Optional, Sequence

from .usfm_parser_state import UsfmParserState
from .usfm_token import UsfmAttribute


class UsfmParserHandler:
    def start_usfm(self, state: UsfmParserState) -> None:
        ...

    def end_usfm(self, state: UsfmParserState) -> None:
        ...

    def got_marker(self, state: UsfmParserState, marker: str) -> None:
        ...

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        ...

    def end_book(self, state: UsfmParserState, marker: str) -> None:
        ...

    def chapter(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        ...

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        ...

    def start_para(
        self, state: UsfmParserState, marker: str, unknown: bool, attributes: Optional[Sequence[UsfmAttribute]]
    ) -> None:
        ...

    def end_para(self, state: UsfmParserState, marker: str) -> None:
        ...

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        ...

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        ...

    def start_note(self, state: UsfmParserState, marker: str, caller: str, category: Optional[str]) -> None:
        ...

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        ...

    def start_table(self, state: UsfmParserState) -> None:
        ...

    def end_table(self, state: UsfmParserState) -> None:
        ...

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        ...

    def end_row(self, state: UsfmParserState, marker: str) -> None:
        ...

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        ...

    def end_cell(self, state: UsfmParserState, marker: str) -> None:
        ...

    def text(self, state: UsfmParserState, text: str) -> None:
        ...

    def unmatched(self, state: UsfmParserState, marker: str) -> None:
        ...

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        ...

    def start_sidebar(self, state: UsfmParserState, marker: str, category: Optional[str]) -> None:
        ...

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        ...

    def opt_break(self, state: UsfmParserState) -> None:
        ...

    def milestone(
        self, state: UsfmParserState, marker: str, start_milestone: bool, attribute: Optional[Sequence[UsfmAttribute]]
    ) -> None:
        ...
