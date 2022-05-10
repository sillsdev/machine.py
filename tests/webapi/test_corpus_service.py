from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from bson.objectid import ObjectId
from mongomock.mongo_client import MongoClient
from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.webapi.corpus_service import CorpusService
from machine.webapi.models import CORPUS_TYPE_TEXT, FILE_FORMAT_PARATEXT, FILE_FORMAT_TEXT, Corpus, DataFile
from machine.webapi.repository import Repository


def test_create_text_corpus() -> None:
    with _TestEnvironment() as env:

        src_corpus1 = env.service.create_text_corpus(env.corpus1_id, "es")
        assert src_corpus1 is not None
        assert [t.id for t in src_corpus1.texts] == ["text1", "text2"]

        src_corpus2 = env.service.create_text_corpus(env.corpus2_id, "es")
        assert src_corpus2 is not None
        assert [t.id for t in src_corpus2.texts] == ["MAT", "MRK"]

        trg_corpus1 = env.service.create_text_corpus(env.corpus1_id, "en")
        assert trg_corpus1 is not None
        assert [t.id for t in trg_corpus1.texts] == ["text1", "text2"]

        trg_corpus2 = env.service.create_text_corpus(env.corpus2_id, "en")
        assert trg_corpus2 is None


class _TestEnvironment(AbstractContextManager):
    def __init__(self) -> None:
        client = MongoClient()
        self._temp_dir = TemporaryDirectory()
        create_test_paratext_backup(Path(self._temp_dir.name))
        self.corpora: Repository[Corpus] = Repository(client.machine.files)
        corpus_ids = self.corpora.insert_many(
            [
                Corpus(
                    owner="owner",
                    name="corpus1",
                    type=CORPUS_TYPE_TEXT,
                    format=FILE_FORMAT_TEXT,
                    files=[
                        DataFile(
                            _id=ObjectId(), languageTag="es", name="src-text1", filename="src-text1.txt", textId="text1"
                        ),
                        DataFile(
                            _id=ObjectId(), languageTag="es", name="src-text2", filename="src-text2.txt", textId="text2"
                        ),
                        DataFile(
                            _id=ObjectId(), languageTag="en", name="trg-text1", filename="trg-text1.txt", textId="text1"
                        ),
                        DataFile(
                            _id=ObjectId(), languageTag="en", name="trg-text2", filename="trg-text2.txt", textId="text2"
                        ),
                    ],
                ),
                Corpus(
                    owner="owner",
                    name="corpus2",
                    type=CORPUS_TYPE_TEXT,
                    format=FILE_FORMAT_PARATEXT,
                    files=[DataFile(_id=ObjectId(), languageTag="es", name="Tes", filename="Tes.zip")],
                ),
            ]
        )
        self.corpus1_id = str(corpus_ids[0])
        self.corpus2_id = str(corpus_ids[1])
        self.service = CorpusService(self.corpora, {"data_files_dir": self._temp_dir.name})

    def __enter__(self) -> "_TestEnvironment":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._temp_dir.cleanup()
