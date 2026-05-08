from typing import Dict, List, Optional

from testutils.memory_paratext_project_file_handler import DefaultParatextProjectSettings
from testutils.memory_paratext_project_quote_convention_detector import MemoryParatextProjectQuoteConventionDetector

from machine.corpora import ParatextProjectSettings
from machine.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    ParatextProjectQuoteConventionDetector,
    QuoteConvention,
    QuoteConventionAnalysis,
)
from machine.scripture import ORIGINAL_VERSIFICATION, get_chapters

standard_english_quote_convention: Optional[QuoteConvention] = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
    "standard_english"
)
standard_french_quote_convention: Optional[QuoteConvention] = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
    "standard_french"
)


def test_get_quote_convention() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id MAT
{get_test_chapter(1, standard_english_quote_convention)}""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention()

    assert analysis.best_quote_convention is not None
    assert analysis.best_quote_convention_score > 0.8
    assert analysis.best_quote_convention.name == "standard_english"


def test_get_quote_convention_by_book() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id MAT
{get_test_chapter(1, standard_english_quote_convention)}""",
            "42MRKTest.SFM": rf"""\id MRK
{get_test_chapter(1, standard_french_quote_convention)}""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention("MRK")

    assert analysis.best_quote_convention is not None
    assert analysis.best_quote_convention_score > 0.8
    assert analysis.best_quote_convention.name == "standard_french"


def test_get_quote_convention_by_chapter() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id MAT
{get_test_chapter(1, standard_english_quote_convention)}""",
            "42MRKTest.SFM": rf"""\id MRK
{get_test_chapter(1, standard_english_quote_convention)}
{get_test_chapter(2, standard_french_quote_convention)}
{get_test_chapter(3, standard_english_quote_convention)}
{get_test_chapter(4, standard_english_quote_convention)}
{get_test_chapter(5, standard_french_quote_convention)}""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention("MRK2,4-5")

    assert analysis.best_quote_convention is not None
    assert analysis.best_quote_convention_score > 0.66
    assert analysis.best_quote_convention.name == "standard_french"


def test_get_quote_convention_by_chapter_indeterminate() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id MAT
{get_test_chapter(1, None)}
{get_test_chapter(2, standard_english_quote_convention)}
{get_test_chapter(3, None)}""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention("MAT1,3")
    assert analysis.best_quote_convention is None


def test_get_quote_convention_invalid_book_code() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id LUK
{get_test_chapter(1, standard_english_quote_convention)}""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention("MAT")
    assert analysis.best_quote_convention is None


def test_get_quote_convention_weighted_average_of_multiple_books() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": rf"""\id MAT
{get_test_chapter(1, standard_english_quote_convention)}""",
            "42MRKTest.SFM": r"""\id MRK
\c 1
\v 1 This "sentence uses a different" convention""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention()

    assert analysis.best_quote_convention is not None
    assert analysis.best_quote_convention.name == "standard_english"
    assert analysis.best_quote_convention_score > 0.8
    assert analysis.best_quote_convention_score < 0.9
    assert (
        analysis.analysis_summary
        == "The most common level 1 quotation marks are “ (5 of 6 opening marks) and ” (5 of 6 closing marks)"
    )


class _TestEnvironment:
    def __init__(
        self,
        settings: Optional[ParatextProjectSettings] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> None:
        self._detector: ParatextProjectQuoteConventionDetector = MemoryParatextProjectQuoteConventionDetector(
            settings or DefaultParatextProjectSettings(), files or {}
        )

    @property
    def detector(self) -> ParatextProjectQuoteConventionDetector:
        return self._detector

    def get_quote_convention(self, scripture_range: Optional[str] = None) -> QuoteConventionAnalysis:
        chapters: Optional[Dict[int, List[int]]] = None
        if scripture_range is not None:
            chapters = get_chapters(scripture_range, ORIGINAL_VERSIFICATION)
        return self.detector.get_quote_convention_analysis(include_chapters=chapters)


def get_test_chapter(number: int, quote_convention: Optional[QuoteConvention]) -> str:
    left_quote: str = quote_convention.get_opening_quotation_mark_at_depth(1) if quote_convention else ""
    right_quote: str = quote_convention.get_closing_quotation_mark_at_depth(1) if quote_convention else ""
    return rf"""\c {number}
\v 1 Someone said, {left_quote}This is something I am saying!
\v 2 This is also something I am saying{right_quote} (that is, {left_quote}something I am speaking{right_quote}).
\p
\v 3 Other text, and someone else said,
\q1
\v 4 {left_quote}Things
\q2 someone else said!
\q3 and more things someone else said.{right_quote}
\m That is why he said {left_quote}things someone else said.{right_quote}
\v 5 Then someone said, {left_quote}More things someone said.{right_quote}"""
