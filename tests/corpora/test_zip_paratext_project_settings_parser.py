from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ContextManager, Optional
from zipfile import ZipFile

from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.corpora import ParatextProjectSettings, UsfmStyleType, UsfmTextType, ZipParatextProjectSettingsParser


def test_parse_custom_stylesheet() -> None:
    with _TestEnvironment() as env:
        settings = env.parser.parse()
        test_tag = settings.stylesheet.get_tag("test")
        assert test_tag.style_type is UsfmStyleType.CHARACTER
        assert test_tag.text_type is UsfmTextType.OTHER


def test_is_daughter_project() -> None:
    with _TestEnvironment() as env:
        settings = env.parser.parse()
        assert settings.has_parent
        assert settings.is_daughter_project_of(settings)
        assert settings.translation_type == "Standard"
        assert not settings.parent_has_been_set

        env.parser = ZipParatextProjectSettingsParser(env.zip_file, settings)

        settings = env.parser.parse()
        assert settings.has_parent
        assert settings.is_daughter_project_of(settings)
        assert settings.translation_type == "Standard"
        assert settings.parent_has_been_set


class _TestEnvironment(ContextManager["_TestEnvironment"]):
    def __init__(self) -> None:
        self._temp_dir = TemporaryDirectory()
        archive_filename = create_test_paratext_backup(Path(self._temp_dir.name))
        self.zip_file = ZipFile(archive_filename)
        self.parser = ZipParatextProjectSettingsParser(self.zip_file)

    def __enter__(self) -> _TestEnvironment:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.zip_file.close()
        self._temp_dir.cleanup()
