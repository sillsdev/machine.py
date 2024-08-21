import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest
from testutils.corpora_test_helpers import TEST_DATA_PATH, USFM_SOURCE_PROJECT_PATH, USFM_TARGET_PROJECT_PATH

from machine.corpora import (
    FileParatextProjectSettingsParser,
    ParatextProjectSettings,
    ParatextTextCorpus,
    ScriptureRef,
    StandardParallelTextCorpus,
    UsfmTextUpdater,
    parse_usfm,
)


@pytest.mark.skip(reason="This is for manual testing only. Remove this decorator to run the test.")
def test_parse_parallel_corpus():
    t_corpus = ParatextTextCorpus(USFM_TARGET_PROJECT_PATH, include_all_text=True, include_markers=True)
    s_corpus = ParatextTextCorpus(USFM_SOURCE_PROJECT_PATH, include_all_text=True, include_markers=True)
    p_corpus = StandardParallelTextCorpus(s_corpus, t_corpus, all_source_rows=True, all_target_rows=False)

    rows = list(p_corpus.get_rows())
    assert rows

    pretranslations: List[Tuple[List[ScriptureRef], str]] = [
        ([ScriptureRef() for s in r.source_refs], r.source_text) for r in rows
    ]

    target_settings = FileParatextProjectSettingsParser(USFM_TARGET_PROJECT_PATH).parse()

    for sfm_file_name in Path(USFM_TARGET_PROJECT_PATH).rglob(
        f"{target_settings.file_name_prefix}*{target_settings.file_name_suffix}"
    ):
        updater = UsfmTextUpdater(pretranslations, strip_all_text=True, prefer_existing_text=False)

        with open(sfm_file_name, mode="r") as sfm_file:
            usfm: str = sfm_file.read()

        parse_usfm(usfm, updater, target_settings.stylesheet, target_settings.versification)
        new_usfm: str = updater.get_usfm(target_settings.stylesheet)
        assert new_usfm is not None


@dataclass
class PretranslationDto:
    text_id: str
    refs: List[str]
    translation: str

    def __post_init__(self):
        if self.text_id is None:
            raise ValueError("text_id is a required field")
        if self.refs is None:
            raise ValueError("refs is a required field")
        if self.translation is None:
            raise ValueError("translation is a required field")


PRETRANSLATION_PATH = TEST_DATA_PATH / "pretranslations.json"
PARATEXT_PROJECT_PATH = TEST_DATA_PATH / "project"


@pytest.mark.skip(reason="This is for manual testing only. Remove this decorator to run the test.")
def test_create_usfm_file():
    parser = FileParatextProjectSettingsParser(PARATEXT_PROJECT_PATH)
    settings: ParatextProjectSettings = parser.parse()

    # Read text from pretranslations file
    with open(PRETRANSLATION_PATH, mode="r") as pretranslation_stream:
        pretranslations_dto: List[PretranslationDto] = [
            PretranslationDto(text_id=item["textId"], refs=item["refs"], translation=item["translation"])
            for item in json.loads(pretranslation_stream.read())
        ]

    pretranslations: List[Tuple[List[ScriptureRef], str]] = [
        (
            [ScriptureRef.parse(ref, settings.versification).to_relaxed() for ref in p.refs] or [],
            p.translation or "",
        )
        for p in pretranslations_dto
    ]

    for sfm_file_name in Path(PARATEXT_PROJECT_PATH).rglob(f"{settings.file_name_prefix}*{settings.file_name_suffix}"):
        updater = UsfmTextUpdater(pretranslations, strip_all_text=True, prefer_existing_text=True)

        with open(sfm_file_name, mode="r") as sfm_file:
            usfm: str = sfm_file.read()

        parse_usfm(usfm, updater, settings.stylesheet, settings.versification)
        new_usfm: str = updater.get_usfm(settings.stylesheet)
        assert new_usfm is not None
