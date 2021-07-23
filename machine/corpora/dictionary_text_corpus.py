from typing import Any, Iterable, Optional, overload

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
        texts: Iterable[Text] = args if len(args) == 0 or isinstance(args[0], Text) else args[0]
        self._texts = {t.id: t for t in texts}

    @property
    def texts(self) -> Iterable[Text]:
        return sorted(self._texts.values(), key=lambda t: t.sort_key)

    def get_text(self, id: str) -> Optional[Text]:
        return self._texts.get(id)

    def get_text_sort_key(self, id: str) -> str:
        return id

    def _add_text(self, text: Text) -> None:
        self._texts[text.id] = text
