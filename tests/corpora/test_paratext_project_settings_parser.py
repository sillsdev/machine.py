from testutils.memory_paratext_project_settings_parser import MemoryParatextProjectSettingsParser


def test_translation_info_empty_values() -> None:
    settings = _create_settings("<TranslationInfo></TranslationInfo>")

    assert settings.translation_type == "Standard"
    assert settings.parent_name is None
    assert settings.parent_guid is None


def test_translation_info_no_parent_specified() -> None:
    settings = _create_settings("<TranslationInfo>BackTranslation::</TranslationInfo>")

    assert settings.translation_type == "BackTranslation"
    assert settings.parent_name is None
    assert settings.parent_guid is None


def test_translation_info_specified() -> None:
    settings = _create_settings("<TranslationInfo>Daughter:DEF:22222222222222222222222222222222</TranslationInfo>")

    assert settings.translation_type == "Daughter"
    assert settings.parent_name == "DEF"
    assert settings.parent_guid == "22222222222222222222222222222222"

def test_normalization_form_default() -> None:
    settings = _create_settings()

    assert settings.normalization_form == "Undefined"


def _create_settings(additional_settings_xml: str = ""):
    files = {
        "Settings.xml": (
            "<ScriptureText>"
            "<Guid>11111111111111111111111111111111</Guid>"
            "<Name>ABC</Name>"
            "<FullName>Test Project</FullName>"
            f"{additional_settings_xml}"
            "</ScriptureText>"
        )
    }

    parser = MemoryParatextProjectSettingsParser(files)
    return parser.parse()
