from pathlib import PurePath
from typing import Iterable, cast, overload

from ..utils.typeshed import StrPath
from .corpora_utils import get_files
from .data_type import DataType
from .dictionary_text_corpus import DictionaryTextCorpus
from .text_file_text import TextFileText


class TextFileTextCorpus(DictionaryTextCorpus):
    @overload
    def __init__(self, file_patterns: Iterable[StrPath], data_types: Iterable[DataType]) -> None: ...

    @overload
    def __init__(self, *file_patterns: StrPath, data_types: Iterable[DataType]) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        file_patterns: Iterable[str]
        if len(args) == 0:
            file_patterns = kwargs.get("file_patterns", [])
        elif isinstance(args[0], str) or isinstance(args[0], PurePath):
            file_patterns = (str(p) for p in args)
        else:
            file_patterns = (str(p) for p in args[0])
        data_types = list(cast(Iterable[DataType], kwargs.get("data_types", [])))
        texts = (
            TextFileText(
                id, filename, data_types[pattern_index] if pattern_index < len(data_types) else DataType.SENTENCE
            )
            for id, filename, pattern_index in get_files(file_patterns)
        )
        super().__init__(texts)
