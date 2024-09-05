from machine.corpora import ParatextProjectSettings, UsfmStylesheet
from machine.scripture import ENGLISH_VERSIFICATION


def test_get_book_file_name_book_num() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("MRK") == "PROJ42.SFM"


def test_get_book_file_name_book_num_book_id() -> None:
    settings = _create_settings("41MAT")
    assert settings.get_book_file_name("MRK") == "PROJ42MRK.SFM"


def test_get_book_file_name_book_id() -> None:
    settings = _create_settings("MAT")
    assert settings.get_book_file_name("MRK") == "PROJMRK.SFM"


def test_get_book_file_name_book_num_double_digit() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("GEN") == "PROJ01.SFM"


def test_get_book_file_name_book_num_xxg() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("XXG") == "PROJ100.SFM"


def test_get_book_file_name_book_num_prefix_a() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("FRT") == "PROJA0.SFM"


def test_get_book_file_name_book_num_prefix_b() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("TDX") == "PROJB0.SFM"


def test_get_book_file_name_book_num_prefix_c() -> None:
    settings = _create_settings("41")
    assert settings.get_book_file_name("3MQ") == "PROJC0.SFM"


def test_get_book_id_book_num() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJ42.SFM") == "MRK"


def test_get_book_id_book_num_book_id() -> None:
    settings = _create_settings("41MAT")
    assert settings.get_book_id("PROJ42MRK.SFM") == "MRK"


def test_get_book_id_book_id() -> None:
    settings = _create_settings("MAT")
    assert settings.get_book_id("PROJMRK.SFM") == "MRK"


def test_get_book_id_book_num_double_digit() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJ01.SFM") == "GEN"


def test_get_book_id_book_num_xxg_book_num() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJ100.SFM") == "XXG"


def test_get_book_id_book_num_xxg_book_num_book_id() -> None:
    settings = _create_settings("41MAT")
    assert settings.get_book_id("PROJ100XXG.SFM") == "XXG"


def test_get_book_id_book_num_prefix_a() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJA0.SFM") == "FRT"


def test_get_book_id_book_num_prefix_b() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJB0.SFM") == "TDX"


def test_get_book_id_book_num_prefix_c() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJC0.SFM") == "3MQ"


def test_get_book_id_wrong_prefix() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("WRONG42.SFM") is None


def test_get_book_id_wrong_suffix() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJ42.WRONG") is None


def test_get_book_id_wrong_book_part_book_num() -> None:
    settings = _create_settings("41")
    assert settings.get_book_id("PROJ42MRK.SFM") is None


def test_get_book_id_wrong_book_part_book_id() -> None:
    settings = _create_settings("MAT")
    assert settings.get_book_id("PROJ42.SFM") is None


def test_get_book_id_wrong_book_part_book_num_book_id() -> None:
    settings = _create_settings("41MAT")
    assert settings.get_book_id("PROJMRK.SFM") is None
    assert settings.get_book_id("PROJ100.SFM") is None


def _create_settings(file_name_form: str) -> ParatextProjectSettings:
    return ParatextProjectSettings(
        "Name",
        "Name",
        "utf-8",
        ENGLISH_VERSIFICATION,
        UsfmStylesheet("usfm.sty"),
        "PROJ",
        file_name_form,
        ".SFM",
        "Major",
        "",
        "BiblicalTerms.xml",
        "en",
    )
