from pathlib import PurePath
from typing import Iterable, cast, overload

from ..utils.typeshed import StrPath
from .corpora_utils import get_files
from .dictionary_text_corpus import DictionaryTextCorpus
from .text_file_text import TextFileText
from .text_row_content_type import TextRowContentType


class TextFileTextCorpus(DictionaryTextCorpus):
    @overload
    def __init__(self, file_patterns: Iterable[StrPath], content_types: Iterable[TextRowContentType] = []) -> None: ...

    @overload
    def __init__(self, *file_patterns: StrPath, content_types: Iterable[TextRowContentType] = []) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        file_patterns: Iterable[str]
        if len(args) == 0:
            file_patterns = kwargs.get("file_patterns", [])
        elif isinstance(args[0], str) or isinstance(args[0], PurePath):
            file_patterns = (str(p) for p in args)
        else:
            file_patterns = (str(p) for p in args[0])
        content_types = list(cast(Iterable[TextRowContentType], kwargs.get("content_types", [])))
        texts = (
            TextFileText(
                id,
                filename,
                content_types[pattern_index] if pattern_index < len(content_types) else TextRowContentType.SEGMENT,
            )
            for id, filename, pattern_index in get_files(file_patterns)
        )
        super().__init__(texts)
