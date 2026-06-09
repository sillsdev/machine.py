from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import FileParatextProjectSettingsParser, UsfmStyleType, UsfmTextType


def test_parse() -> None:
    parser = FileParatextProjectSettingsParser(USFM_TEST_PROJECT_PATH)
    settings = parser.parse()
    assert settings.guid == "a7e0b3ce0200736062f9f810a444dbfbe64aca35"
    assert settings.name == "Tes"
    assert settings.full_name == "Test"
    assert settings.encoding == "utf_8_sig"
    assert settings.versification.name.startswith("English")
    assert settings.file_name_prefix == ""
    assert settings.file_name_form == "41MAT"
    assert settings.file_name_suffix == "Tes.SFM"
    assert settings.biblical_terms_list_type == "Project"
    assert settings.biblical_terms_project_name == "Tes"
    assert settings.biblical_terms_file_name == "ProjectBiblicalTerms.xml"
    assert settings.language_code == "en"
    assert settings.translation_type == "Standard"
    assert settings.visibility == "Public"


def test_parse_custom_stylesheet() -> None:
    parser = FileParatextProjectSettingsParser(USFM_TEST_PROJECT_PATH)
    settings = parser.parse()
    test_tag = settings.stylesheet.get_tag("test")
    assert test_tag.style_type is UsfmStyleType.CHARACTER
    assert test_tag.text_type is UsfmTextType.OTHER


def test_is_daughter_project() -> None:
    parser = FileParatextProjectSettingsParser(USFM_TEST_PROJECT_PATH)
    settings = parser.parse()
    assert settings.translation_type == "Standard"
    assert settings.visibility == "Public"
    assert settings.has_parent
    assert settings.is_daughter_project_of(settings)
    assert settings.parent is None

    parser = FileParatextProjectSettingsParser(USFM_TEST_PROJECT_PATH, settings)
    settings = parser.parse()
    assert settings.has_parent
    assert settings.is_daughter_project_of(settings)
    assert settings.parent is not None
