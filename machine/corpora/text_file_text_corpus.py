from pathlib import PurePath
from typing import Any, Iterable, overload

from ..tokenization.tokenizer import Tokenizer
from ..utils.typeshed import StrPath
from .corpora_helpers import get_files
from .dictionary_text_corpus import DictionaryTextCorpus
from .text_file_text import TextFileText


class TextFileTextCorpus(DictionaryTextCorpus):
    @overload
    def __init__(self, word_tokenizer: Tokenizer[str, int, str], file_patterns: Iterable[StrPath]) -> None:
        ...

    @overload
    def __init__(self, word_tokenizer: Tokenizer[str, int, str], *file_patterns: StrPath) -> None:
        ...

    def __init__(self, word_tokenizer: Tokenizer[str, int, str], *args: Any) -> None:
        file_patterns: Iterable[str]
        if len(args) == 0:
            file_patterns = []
        elif isinstance(args[0], str) or isinstance(args[0], PurePath):
            file_patterns = (str(p) for p in args)
        else:
            file_patterns = (str(p) for p in args[0])
        texts = (TextFileText(word_tokenizer, id, filename) for id, filename in get_files(file_patterns))
        super().__init__(texts)
