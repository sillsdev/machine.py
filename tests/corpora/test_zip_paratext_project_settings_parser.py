from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ContextManager
from zipfile import ZipFile

from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.corpora import UsfmStyleType, UsfmTextType, ZipParatextProjectSettingsParser


def test_parse_custom_stylesheet() -> None:
    with _TestEnvironment() as env:
        settings = env.parser.parse()
        test_tag = settings.stylesheet.get_tag("test")
        assert test_tag.style_type is UsfmStyleType.CHARACTER
        assert test_tag.text_type is UsfmTextType.OTHER


class _TestEnvironment(ContextManager["_TestEnvironment"]):
    def __init__(self) -> None:
        self._temp_dir = TemporaryDirectory()
        archive_filename = create_test_paratext_backup(Path(self._temp_dir.name))
        self._zip_file = ZipFile(archive_filename)
        self._parser = ZipParatextProjectSettingsParser(self._zip_file)

    @property
    def parser(self) -> ZipParatextProjectSettingsParser:
        return self._parser

    def __enter__(self) -> _TestEnvironment:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._zip_file.close()
        self._temp_dir.cleanup()
