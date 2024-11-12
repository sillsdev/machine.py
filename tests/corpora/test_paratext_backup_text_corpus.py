from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ContextManager

from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.corpora import ParatextBackupTextCorpus


def test_texts() -> None:
    with _TestEnvironment() as env:
        assert [t.id for t in env.corpus.texts] == ["LEV", "1CH", "MAT", "MRK", "JHN"]


def test_get_text() -> None:
    with _TestEnvironment() as env:
        mat = env.corpus.get_text("MAT")
        assert mat is not None
        assert any(mat.get_rows())

        luk = env.corpus.get_text("LUK")
        assert luk is None

        jhn = env.corpus.get_text("JHN")
        assert jhn is not None
        assert not any(jhn.get_rows())


class _TestEnvironment(ContextManager["_TestEnvironment"]):
    def __init__(self) -> None:
        self._temp_dir = TemporaryDirectory()
        archive_filename = create_test_paratext_backup(Path(self._temp_dir.name))
        self._corpus = ParatextBackupTextCorpus(archive_filename)

    @property
    def corpus(self) -> ParatextBackupTextCorpus:
        return self._corpus

    def __enter__(self) -> _TestEnvironment:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._temp_dir.cleanup()
