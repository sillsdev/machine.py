from typing import Any, Dict, Iterable, Optional, overload

from .text import Text
from .text_corpus import TextCorpus


class DictionaryTextCorpus(TextCorpus):
    @overload
    def __init__(self, *args: Text) -> None:
        ...

    @overload
    def __init__(self, texts: Iterable[Text]) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        self._text_dictionary: Dict[str, Text]
        if len(args) == 0:
            self._text_dictionary = {}
        elif isinstance(args[0], Text):
            self._text_dictionary = {t.id: t for t in args}
        else:
            self._text_dictionary = {t.id: t for t in args[0]}

    @property
    def texts(self) -> Iterable[Text]:
        return sorted(self._text_dictionary.values(), key=lambda t: t.sort_key)

    def get_text(self, id: str) -> Optional[Text]:
        return self._text_dictionary.get(id)

    def get_text_sort_key(self, id: str) -> str:
        return id

    def _add_text(self, text: Text) -> None:
        self._text_dictionary[text.id] = text
