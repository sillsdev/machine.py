from typing import Generator, Iterable, Optional, cast

from ..scripture.verse_ref import Versification
from .dictionary_text_corpus import DictionaryTextCorpus
from .scripture_text import ScriptureText
from .text_corpus_row import TextCorpusRow
from .text_corpus_view import TextCorpusView


class ScriptureTextCorpus(DictionaryTextCorpus):
    def __init__(self, versification: Versification, texts: Iterable[ScriptureText] = []) -> None:
        super().__init__(texts)
        self._versification = versification

    @property
    def versification(self) -> Versification:
        return self._versification

    def _get_rows(self, based_on: Optional[TextCorpusView]) -> Generator[TextCorpusRow, None, None]:
        based_on_versification: Optional[Versification] = None
        if based_on is not None and isinstance(based_on.source, ScriptureTextCorpus):
            based_on_versification = based_on.source.versification
        for text in cast(Iterable[ScriptureText], self.texts):
            with text.get_rows(based_on_versification) as rows:
                yield from rows
