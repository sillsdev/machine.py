from pathlib import PurePath
from typing import Iterable, overload

from ..utils.typeshed import StrPath
from .corpora_utils import get_files
from .dictionary_alignment_corpus import DictionaryAlignmentCorpus
from .text_file_alignment_collection import TextFileAlignmentCollection


class TextFileAlignmentCorpus(DictionaryAlignmentCorpus):
    @overload
    def __init__(self, file_patterns: Iterable[StrPath]) -> None:
        ...

    @overload
    def __init__(self, *file_patterns: StrPath) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        file_patterns: Iterable[str]
        if len(args) == 0:
            file_patterns = kwargs.get("file_patterns", [])
        elif isinstance(args[0], str) or isinstance(args[0], PurePath):
            file_patterns = (str(p) for p in args)
        else:
            file_patterns = (str(p) for p in args[0])
        collections = (TextFileAlignmentCollection(id, filename) for id, filename in get_files(file_patterns))
        super().__init__(collections)
