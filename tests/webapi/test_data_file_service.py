from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from bson.objectid import ObjectId
from mongomock.mongo_client import MongoClient
from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.webapi import DataFileService, Repository
from machine.webapi.models import (
    CORPUS_TYPE_SOURCE,
    CORPUS_TYPE_TARGET,
    DATA_TYPE_TEXT_CORPUS,
    FILE_FORMAT_PARATEXT,
    FILE_FORMAT_TEXT,
    DataFile,
)


def test_create_text_corpora() -> None:
    with _TestEnvironment() as env:
        src_corpora = env.service.create_text_corpora(str(env.engine_id), CORPUS_TYPE_SOURCE)
        assert len(src_corpora) == 2
        src_corpus1 = src_corpora["corpus1"]
        assert [t.id for t in src_corpus1.texts] == ["text1", "text2"]
        src_corpus2 = src_corpora["corpus2"]
        assert [t.id for t in src_corpus2.texts] == ["MAT", "MRK"]

        trg_corpora = env.service.create_text_corpora(str(env.engine_id), CORPUS_TYPE_TARGET)
        assert len(trg_corpora) == 1
        trg_corpus1 = trg_corpora["corpus1"]
        assert [t.id for t in trg_corpus1.texts] == ["text1", "text2"]


def test_get_texts_to_translate() -> None:
    with _TestEnvironment() as env:
        corpora_translate_text_ids = env.service.get_texts_to_translate(str(env.engine_id))

        corpus1_translate_text_ids = corpora_translate_text_ids["corpus1"]
        assert corpus1_translate_text_ids == {"text2"}

        corpus2_translate_text_ids = corpora_translate_text_ids["corpus2"]
        assert corpus2_translate_text_ids == {"MAT", "MRK"}


class _TestEnvironment(AbstractContextManager):
    def __init__(self) -> None:
        client = MongoClient()
        self._temp_dir = TemporaryDirectory()
        create_test_paratext_backup(Path(self._temp_dir.name))
        self.engine_id = ObjectId()
        self.files: Repository[DataFile] = Repository(client.machine.files)
        self.files.insert_many(
            [
                DataFile(
                    engineRef=self.engine_id,
                    dataType=DATA_TYPE_TEXT_CORPUS,
                    name="text1",
                    format=FILE_FORMAT_TEXT,
                    corpusType=CORPUS_TYPE_SOURCE,
                    corpusId="corpus1",
                    filename="src-text1.txt",
                ),
                DataFile(
                    engineRef=self.engine_id,
                    dataType=DATA_TYPE_TEXT_CORPUS,
                    name="text2",
                    format=FILE_FORMAT_TEXT,
                    corpusType=CORPUS_TYPE_SOURCE,
                    corpusId="corpus1",
                    translate=True,
                    filename="src-text2.txt",
                ),
                DataFile(
                    engineRef=self.engine_id,
                    dataType=DATA_TYPE_TEXT_CORPUS,
                    name="text1",
                    format=FILE_FORMAT_TEXT,
                    corpusType=CORPUS_TYPE_TARGET,
                    corpusId="corpus1",
                    filename="trg-text1.txt",
                ),
                DataFile(
                    engineRef=self.engine_id,
                    dataType=DATA_TYPE_TEXT_CORPUS,
                    name="text2",
                    format=FILE_FORMAT_TEXT,
                    corpusType=CORPUS_TYPE_TARGET,
                    corpusId="corpus1",
                    filename="trg-text2.txt",
                ),
                DataFile(
                    engineRef=self.engine_id,
                    dataType=DATA_TYPE_TEXT_CORPUS,
                    name="src-project",
                    format=FILE_FORMAT_PARATEXT,
                    corpusType=CORPUS_TYPE_SOURCE,
                    corpusId="corpus2",
                    translate=True,
                    filename="Tes.zip",
                ),
            ]
        )
        self.service = DataFileService(self.files, {"data_files_dir": self._temp_dir.name})

    def __enter__(self) -> "_TestEnvironment":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._temp_dir.cleanup()
