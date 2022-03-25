from typing import Iterable, Optional, overload

from .text import Text
from .text_corpus import TextCorpus


class DictionaryTextCorpus(TextCorpus):
    @overload
    def __init__(self, *texts: Text) -> None:
        ...

    @overload
    def __init__(self, texts: Iterable[Text]) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        texts: Iterable[Text]
        if len(args) == 0:
            texts = kwargs.get("texts", [])
        elif isinstance(args[0], Text):
            texts = args
        else:
            texts = args[0]
        self._texts = {t.id: t for t in texts}

    @property
    def texts(self) -> Iterable[Text]:
        return sorted(self._texts.values(), key=lambda t: t.sort_key)

    def __getitem__(self, id: str) -> Optional[Text]:
        return self._texts.get(id)

    def get_text(self, id: str) -> Optional[Text]:
        return self[id]

    def _add_text(self, text: Text) -> None:
        self._texts[text.id] = text
