import zipfile
from pathlib import Path
from typing import List, Optional

import pytest
from testutils.corpora_test_helpers import (
    USFM_SOURCE_PROJECT_PATH,
    USFM_SOURCE_PROJECT_ZIP_PATH,
    USFM_TARGET_PROJECT_PATH,
    USFM_TARGET_PROJECT_ZIP_PATH,
)

from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    ParatextTextCorpus,
    ScriptureRef,
    StandardParallelTextCorpus,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
)
from machine.corpora.zip_paratext_project_versification_detector import ZipParatextProjectVersificationErrorDetector
from machine.punctuation_analysis import QuoteConventionDetector, ZipParatextProjectQuoteConventionDetector


@pytest.mark.skip(reason="This is for manual testing only. Remove this decorator to run the test.")
def test_parse_parallel_corpus():
    t_corpus = ParatextTextCorpus(USFM_TARGET_PROJECT_PATH, include_all_text=True, include_markers=True)
    s_corpus = ParatextTextCorpus(USFM_SOURCE_PROJECT_PATH, include_all_text=True, include_markers=True)
    p_corpus = StandardParallelTextCorpus(s_corpus, t_corpus, all_source_rows=True, all_target_rows=False)

    rows = list(p_corpus.get_rows())
    assert rows

    pretranslations: List[UpdateUsfmRow] = [
        (UpdateUsfmRow(refs=[ScriptureRef.parse(s) for s in r.source_refs], text=r.source_text)) for r in rows
    ]

    target_settings = FileParatextProjectSettingsParser(USFM_TARGET_PROJECT_PATH).parse()
    updater = FileParatextProjectTextUpdater(USFM_TARGET_PROJECT_PATH)
    for sfm_file in Path(USFM_TARGET_PROJECT_PATH).rglob(
        f"{target_settings.file_name_prefix}*{target_settings.file_name_suffix}"
    ):
        sfm_file_name: str = sfm_file.name
        book_id: Optional[str] = target_settings.get_book_id(sfm_file_name)
        if not book_id:
            continue
        new_usfm: Optional[str] = updater.update_usfm(
            book_id, pretranslations, text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING
        )
        assert new_usfm is not None


@pytest.mark.skip(reason="This is for manual testing only. Remove this decorator to run the test.")
def test_analyze_corpora_quote_conventions():
    source_handler = QuoteConventionDetector()
    source_archive = zipfile.ZipFile(USFM_SOURCE_PROJECT_ZIP_PATH, "r")
    source_quote_convention_detector = ZipParatextProjectQuoteConventionDetector(source_archive)
    source_quote_convention_detector.get_quote_convention_analysis(source_handler)

    target_handler = QuoteConventionDetector()
    target_archive = zipfile.ZipFile(USFM_TARGET_PROJECT_ZIP_PATH, "r")
    target_quote_convention_detector = ZipParatextProjectQuoteConventionDetector(target_archive)
    target_quote_convention_detector.get_quote_convention_analysis(target_handler)

    source_analysis = source_handler.detect_quote_convention()
    target_analysis = target_handler.detect_quote_convention()

    assert source_analysis is not None
    assert target_analysis is not None


@pytest.mark.skip(reason="This is for manual testing only. Remove this decorator to run the test.")
def test_validate_usfm_versification():
    archive = zipfile.ZipFile(USFM_SOURCE_PROJECT_ZIP_PATH, "r")
    versification_error_detector = ZipParatextProjectVersificationErrorDetector(archive)
    errors = versification_error_detector.get_usfm_versification_errors()
    assert len(errors) == 0
