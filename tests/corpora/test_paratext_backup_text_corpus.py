from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ContextManager, Optional

from pytest import raises
from testutils.corpora_test_helpers import (
    create_test_paratext_backup,
    create_test_paratext_backup_invalid_id,
    create_test_paratext_backup_mismatch_id,
)

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


def test_invalid_id() -> None:
    with raises(ValueError, match=r"The \\id tag in .* is invalid."):
        with _TestEnvironment("invalid_id") as env:
            env.corpus.get_text("JDG")


def test_mismatch_id() -> None:
    with raises(ValueError, match=r"The \\id tag .* in .* does not match filename book id .*"):
        with _TestEnvironment("mismatch_id") as env:
            env.corpus.get_text("JDG")


class _TestEnvironment(ContextManager["_TestEnvironment"]):
    def __init__(self, project_folder_name: Optional[str] = None) -> None:
        self._temp_dir = TemporaryDirectory()
        if project_folder_name == "invalid_id":
            archive_filename = create_test_paratext_backup_invalid_id(Path(self._temp_dir.name))
        elif project_folder_name == "mismatch_id":
            archive_filename = create_test_paratext_backup_mismatch_id(Path(self._temp_dir.name))
        else:
            archive_filename = create_test_paratext_backup(Path(self._temp_dir.name))
        self._corpus = ParatextBackupTextCorpus(archive_filename)

    @property
    def corpus(self) -> ParatextBackupTextCorpus:
        return self._corpus

    def __enter__(self) -> _TestEnvironment:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._temp_dir.cleanup()
