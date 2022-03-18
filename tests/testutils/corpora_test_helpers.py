import shutil
from pathlib import Path

from machine.corpora import TextCorpusRow
from machine.scripture import VerseRef

from . import TEST_DATA_PATH

USFM_TEST_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "Tes"
USX_TEST_PROJECT_PATH = TEST_DATA_PATH / "usx" / "Tes"


def create_test_dbl_bundle(temp_dir: Path) -> Path:
    shutil.make_archive(str(temp_dir / "Tes"), "zip", USX_TEST_PROJECT_PATH)
    return temp_dir / "Tes.zip"


def verse_ref(segment: TextCorpusRow) -> VerseRef:
    assert isinstance(segment.ref, VerseRef)
    return segment.ref
