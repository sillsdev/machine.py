from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import FileParatextProjectSettingsParser, UsfmStyleType, UsfmTextType


def test_parse_custom_stylesheet() -> None:
    parser = FileParatextProjectSettingsParser(USFM_TEST_PROJECT_PATH)
    settings = parser.parse()
    test_tag = settings.stylesheet.get_tag("test")
    assert test_tag.style_type is UsfmStyleType.CHARACTER
    assert test_tag.text_type is UsfmTextType.OTHER
