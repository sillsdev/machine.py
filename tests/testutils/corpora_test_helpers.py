import shutil
from pathlib import Path

from machine.corpora.text_segment import TextSegment
from machine.scripture.verse_ref import VerseRef

from . import TEST_DATA_PATH

USFM_TEST_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "Tes"
USX_TEST_PROJECT_PATH = TEST_DATA_PATH / "usx" / "Tes"


def create_test_dbl_bundle(temp_dir: Path) -> Path:
    shutil.make_archive(str(temp_dir / "Tes"), "zip", USX_TEST_PROJECT_PATH)
    return temp_dir / "Tes.zip"


def verse_ref(segment: TextSegment) -> VerseRef:
    assert isinstance(segment.segment_ref, VerseRef)
    return segment.segment_ref
