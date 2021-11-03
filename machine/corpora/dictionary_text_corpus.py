from typing import Any, Iterable, overload

from .null_text import NullText
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

    def __getitem__(self, id: str) -> Text:
        text = self._texts.get(id)
        if text is None:
            text = self.create_null_text(id)
        return text

    def create_null_text(self, id: str) -> Text:
        return NullText(id, id)

    def _add_text(self, text: Text) -> None:
        self._texts[text.id] = text
