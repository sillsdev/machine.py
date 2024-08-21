import shutil
from pathlib import Path

from machine.corpora import ScriptureRef, TextRow
from machine.scripture import VerseRef

from . import TEST_DATA_PATH

USFM_TEST_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "Tes"
USFM_TARGET_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "target"
USFM_SOURCE_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "source"
USX_TEST_PROJECT_PATH = TEST_DATA_PATH / "usx" / "Tes"
TEXT_TEST_PROJECT_PATH = TEST_DATA_PATH / "txt"
CUSTOM_VERS_PATH = TEST_DATA_PATH / "custom.vrs"


def create_test_dbl_bundle(temp_dir: Path) -> Path:
    shutil.make_archive(str(temp_dir / "Tes"), "zip", USX_TEST_PROJECT_PATH)
    return temp_dir / "Tes.zip"


def create_test_paratext_backup(temp_dir: Path) -> Path:
    shutil.make_archive(str(temp_dir / "Tes"), "zip", USFM_TEST_PROJECT_PATH)
    return temp_dir / "Tes.zip"


def verse_ref(segment: TextRow) -> VerseRef:
    assert isinstance(segment.ref, VerseRef)
    return segment.ref


def scripture_ref(segment: TextRow) -> ScriptureRef:
    assert isinstance(segment.ref, ScriptureRef)
    return segment.ref
