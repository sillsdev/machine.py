import shutil
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parent / "test_data"
USFM_STYLESHEET_PATH = TEST_DATA_PATH / "usfm" / "usfm.sty"
USFM_TEST_PROJECT_PATH = TEST_DATA_PATH / "usfm" / "Tes"
USX_TEST_PROJECT_PATH = TEST_DATA_PATH / "usx" / "Tes"


def create_test_dbl_bundle(temp_dir: Path) -> Path:
    shutil.make_archive(str(temp_dir / "Tes"), "zip", USX_TEST_PROJECT_PATH)
    return temp_dir / "Tes.zip"
