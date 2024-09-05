import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from testutils.corpora_test_helpers import TEST_DATA_PATH, USFM_SOURCE_PROJECT_PATH, USFM_TARGET_PROJECT_PATH

from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    ParatextTextCorpus,
    ScriptureRef,
    StandardParallelTextCorpus,
    ZipParatextProjectSettingsParser,
    ZipParatextProjectTextUpdater,
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
    updater = FileParatextProjectTextUpdater(USFM_TARGET_PROJECT_PATH)
    for sfm_file in Path(USFM_TARGET_PROJECT_PATH).rglob(
        f"{target_settings.file_name_prefix}*{target_settings.file_name_suffix}"
    ):
        sfm_file_name: str = sfm_file.name
        book_id: Optional[str] = target_settings.get_book_id(sfm_file_name)
        if not book_id:
            continue
        new_usfm: Optional[str] = updater.update_usfm(
            book_id, pretranslations, strip_all_text=True, prefer_existing_text=False
        )
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
# In order to run this test on specific projects, place the Paratext projects or Paratext project zips in the
# tests/testutils/data/project/ folder.
def test_create_usfm_file():
    def get_usfm(project_path: Path):
        project_archive = None
        try:
            project_archive = zipfile.ZipFile(project_path, "r")
            parser = ZipParatextProjectSettingsParser(project_archive)
        except IsADirectoryError:
            parser = FileParatextProjectSettingsParser(project_path)

        settings = parser.parse()

        # Read text from pretranslations file
        with open(PRETRANSLATION_PATH, "r") as pretranslation_stream:
            pretranslations = [
                (
                    [ScriptureRef.parse(r, settings.versification).to_relaxed() for r in p["refs"] or []],
                    p.get("translation", ""),
                )
                for p in json.load(pretranslation_stream)
            ]

        book_ids: List[str] = []
        if project_archive is None:
            for sfm_file in Path(project_path).glob(f"{settings.file_name_prefix}*{settings.file_name_suffix}"):
                book_id = settings.get_book_id(sfm_file.name)
                if book_id:
                    book_ids.append(book_id)
            updater = FileParatextProjectTextUpdater(project_path)
        else:
            for entry in project_archive.infolist():
                if entry.filename.startswith(settings.file_name_prefix) and entry.filename.endswith(
                    settings.file_name_suffix
                ):
                    book_id = settings.get_book_id(entry.filename)
                    if book_id:
                        book_ids.append(book_id)
            updater = ZipParatextProjectTextUpdater(project_archive)

        for book_id in book_ids:
            new_usfm = updater.update_usfm(book_id, pretranslations, strip_all_text=True, prefer_existing_text=False)
            assert new_usfm is not None

    if not Path(PARATEXT_PROJECT_PATH / "Settings.xml").exists():
        for subdir in PARATEXT_PROJECT_PATH.iterdir():
            try:
                get_usfm(subdir)
            except Exception as e:
                assert False, f"Failed to process {subdir}: {e}"
    else:
        get_usfm(PARATEXT_PROJECT_PATH)
