from typing import Generator, Iterable, Optional, overload

from ..scripture.verse_ref import Versification
from .text import Text
from .text_corpus import TextCorpus
from .text_row import TextRow


class DictionaryTextCorpus(TextCorpus):
    @overload
    def __init__(self, *texts: Text) -> None: ...

    @overload
    def __init__(self, texts: Iterable[Text]) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        texts: Iterable[Text]
        if len(args) == 0:
            texts = kwargs.get("texts", [])
        elif isinstance(args[0], Text):
            texts = args
        else:
            texts = args[0]
        self._texts = {t.id: t for t in texts}
        self._is_tokenized = False
        self._versification = None

    @property
    def texts(self) -> Iterable[Text]:
        return sorted(self._texts.values(), key=lambda t: t.sort_key)

    @property
    def is_tokenized(self) -> bool:
        return self._is_tokenized

    @is_tokenized.setter
    def is_tokenized(self, value: bool) -> None:
        self._is_tokenized = value

    @property
    def versification(self) -> Optional[Versification]:
        return self._versification

    @versification.setter
    def versification(self, value: Versification) -> None:
        self._versification = value

    def count(self, include_empty: bool = True, text_ids: Optional[Iterable[str]] = None) -> int:
        texts = self._texts.values()
        if text_ids is not None:
            texts = (t for t in texts if t.id in text_ids)
        return sum(t.count(include_empty) for t in texts)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        texts = self.texts
        if text_ids is not None:
            texts = (t for t in texts if t.id in text_ids)
        for t in texts:
            yield from t.get_rows()

    def __getitem__(self, id: str) -> Optional[Text]:
        return self._texts.get(id)

    def get_text(self, id: str) -> Optional[Text]:
        return self[id]

    def _add_text(self, text: Text) -> None:
        self._texts[text.id] = text
