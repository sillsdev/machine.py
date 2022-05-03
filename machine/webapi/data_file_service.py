from itertools import groupby
from pathlib import Path
from typing import Dict, Set

from bson.objectid import ObjectId

from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.paratext_backup_text_corpus import ParatextBackupTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text import TextFileText
from .models import CORPUS_TYPE_SOURCE, DATA_TYPE_TEXT_CORPUS, FILE_FORMAT_PARATEXT, FILE_FORMAT_TEXT, DataFile
from .repository import Repository


class DataFileService:
    def __init__(self, data_files: Repository[DataFile], config: dict) -> None:
        self._data_files = data_files
        self._config = config

    def create_text_corpora(self, engine_id: str, corpus_type: str) -> Dict[str, TextCorpus]:
        cursor = self._data_files.get_all(
            {
                "engineRef": ObjectId(engine_id),
                "dataType": DATA_TYPE_TEXT_CORPUS,
                "corpusType": corpus_type,
                "corpusId": {"$ne": None},
            }
        )

        corpora: Dict[str, TextCorpus] = {}
        data_files = sorted(cursor, key=lambda data_file: data_file.get("corpusId", ""))
        for key, grouping in groupby(data_files, lambda data_file: data_file.get("corpusId", "")):
            corpus_data_files = list(grouping)
            format = corpus_data_files[0]["format"]
            if format == FILE_FORMAT_TEXT:
                corpus = DictionaryTextCorpus(
                    TextFileText(f["name"], self._get_data_file_path(f)) for f in corpus_data_files
                )
            elif format == FILE_FORMAT_PARATEXT:
                corpus = ParatextBackupTextCorpus(self._get_data_file_path(corpus_data_files[0]))
            else:
                raise RuntimeError(f"Unknown format: {format}")

            corpora[key] = corpus

        return corpora

    def get_texts_to_translate(self, engine_id: str) -> Dict[str, Set[str]]:
        cursor = self._data_files.get_all(
            {
                "engineRef": ObjectId(engine_id),
                "dataType": DATA_TYPE_TEXT_CORPUS,
                "corpusType": CORPUS_TYPE_SOURCE,
                "corpusId": {"$ne": None},
                "translate": True,
            }
        )

        corpora_translate_text_ids: Dict[str, Set[str]] = {}
        data_files = sorted(cursor, key=lambda data_file: data_file.get("corpusId", ""))
        for key, grouping in groupby(data_files, lambda data_file: data_file.get("corpusId", "")):
            corpus_data_files = list(grouping)
            format = corpus_data_files[0]["format"]
            if format == FILE_FORMAT_TEXT:
                translate_text_ids = {f["name"] for f in corpus_data_files}
            elif format == FILE_FORMAT_PARATEXT:
                corpus = ParatextBackupTextCorpus(self._get_data_file_path(corpus_data_files[0]))
                translate_text_ids = {t.id for t in corpus.texts}
            else:
                raise RuntimeError(f"Unknown format: {format}")
            corpora_translate_text_ids[key] = translate_text_ids

        return corpora_translate_text_ids

    def _get_data_file_path(self, data_file: DataFile) -> Path:
        return Path(self._config["data_files_dir"], data_file["filename"])
