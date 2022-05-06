from pathlib import Path
from typing import Optional

from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.paratext_backup_text_corpus import ParatextBackupTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text import TextFileText
from .models import CORPUS_TYPE_TEXT, FILE_FORMAT_PARATEXT, FILE_FORMAT_TEXT, Corpus, DataFile
from .repository import Repository


class CorpusService:
    def __init__(self, corpora: Repository[Corpus], config: dict) -> None:
        self._corpora = corpora
        self._config = config

    def create_text_corpus(self, id: str, language_tag: str) -> Optional[TextCorpus]:
        corpus = self._corpora.get(id)
        if corpus is None or corpus["type"] != CORPUS_TYPE_TEXT:
            return None

        files = [f for f in corpus["files"] if f["languageTag"] == language_tag]
        if len(files) == 0:
            return None

        text_corpus: Optional[TextCorpus] = None
        format = corpus["format"]
        if format == FILE_FORMAT_TEXT:
            text_corpus = DictionaryTextCorpus(
                TextFileText(f.get("textId", f["name"]), self._get_data_file_path(f)) for f in files
            )
        elif format == FILE_FORMAT_PARATEXT:
            text_corpus = ParatextBackupTextCorpus(self._get_data_file_path(files[0]))

        return text_corpus

    def _get_data_file_path(self, data_file: DataFile) -> Path:
        return Path(self._config["data_files_dir"], data_file["filename"])
