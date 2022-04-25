import os
from itertools import groupby
from typing import Dict, Optional

from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.paratext_text_corpus import ParatextTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text import TextFileText
from .models import DATA_TYPE_TEXT_CORPUS, FILE_FORMAT_PARATEXT, FILE_FORMAT_TEXT, DataFile
from .repository import Repository


class DataFileService:
    def __init__(self, data_files: Repository[DataFile], config: dict) -> None:
        self._data_files = data_files
        self._config = config

    def create_text_corpora(self, engine_id: str, corpus_type: str) -> Dict[str, TextCorpus]:
        cursor = self._data_files.get_all(
            {
                "engineRef": engine_id,
                "dataType": DATA_TYPE_TEXT_CORPUS,
                "corpusType": corpus_type,
            }
        )

        corpora: Dict[str, TextCorpus] = {}
        data_files = sorted(cursor, key=lambda data_file: data_file.get("corpusId", ""))
        for key, grouping in groupby(data_files, lambda data_file: data_file.get("corpusId", "")):
            if key == "":
                continue
            corpus_data_files = list(grouping)
            format = corpus_data_files[0].get("format", FILE_FORMAT_TEXT)
            corpus: Optional[TextCorpus] = None
            if format == FILE_FORMAT_TEXT:
                corpus = DictionaryTextCorpus(
                    TextFileText(f.get("name", ""), self._get_data_file_path(f)) for f in corpus_data_files
                )
            elif format == FILE_FORMAT_PARATEXT:
                corpus = ParatextTextCorpus(self._get_data_file_path(corpus_data_files[0]))

            if corpus is not None:
                corpora[key] = corpus

        return corpora

    def _get_data_file_path(self, data_file: DataFile) -> str:
        return os.path.join(self._config["data_files_dir"], data_file.get("filename", ""))
