from typing import Dict, Optional

from testutils.memory_paratext_project_quote_convention_detector import MemoryParatextProjectQuoteConventionDetector

from machine.corpora import ParatextProjectSettings, UsfmStylesheet
from machine.punctuation_analysis.paratext_project_quote_convention_detector import (
    ParatextProjectQuoteConventionDetector,
)
from machine.punctuation_analysis.quote_convention_detector import QuoteConventionAnalysis
from machine.scripture import ORIGINAL_VERSIFICATION, Versification


def test_get_quote_convention() -> None:
    env = _TestEnvironment(
        files={
            "41MATTest.SFM": r"""\id MAT
\c 1
\v 1 Someone said, “This is something I am saying!
\v 2 This is also something I am saying” (that is, “something I am speaking”).
\p
\v 3 Other text, and someone else said,
\q1
\v 4 “Things
\q2 someone else said!
\q3 and more things someone else said.”
\m That is why he said “things someone else said.”
\v 5 Then someone said, “More things someone said.”""",
        }
    )
    analysis: Optional[QuoteConventionAnalysis] = env.get_quote_convention()
    assert analysis is not None
    assert analysis.best_quote_convention_score > 0.8
    assert analysis.best_quote_convention.name == "standard_english"


class _TestEnvironment:
    def __init__(
        self,
        settings: Optional[ParatextProjectSettings] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> None:
        self._detector: ParatextProjectQuoteConventionDetector = MemoryParatextProjectQuoteConventionDetector(
            settings or _DefaultParatextProjectSettings(), files or {}
        )

    @property
    def detector(self) -> ParatextProjectQuoteConventionDetector:
        return self._detector

    def get_quote_convention(self) -> Optional[QuoteConventionAnalysis]:
        return self.detector.get_quote_convention_analysis()


class _DefaultParatextProjectSettings(ParatextProjectSettings):
    def __init__(
        self,
        name: str = "Test",
        full_name: str = "TestProject",
        encoding: Optional[str] = None,
        versification: Optional[Versification] = None,
        stylesheet: Optional[UsfmStylesheet] = None,
        file_name_prefix: str = "",
        file_name_form: str = "41MAT",
        file_name_suffix: str = "Test.SFM",
        biblical_terms_list_type: str = "Project",
        biblical_terms_project_name: str = "Test",
        biblical_terms_file_name: str = "ProjectBiblicalTerms.xml",
        language_code: str = "en",
    ):

        super().__init__(
            name,
            full_name,
            encoding if encoding is not None else "utf-8",
            versification if versification is not None else ORIGINAL_VERSIFICATION,
            stylesheet if stylesheet is not None else UsfmStylesheet("usfm.sty"),
            file_name_prefix,
            file_name_form,
            file_name_suffix,
            biblical_terms_list_type,
            biblical_terms_project_name,
            biblical_terms_file_name,
            language_code,
        )
