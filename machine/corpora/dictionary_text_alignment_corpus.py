from typing import Any, Generator, Iterable, Optional, overload

from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus import TextAlignmentCorpus
from .text_alignment_corpus_row import TextAlignmentCorpusRow


class DictionaryTextAlignmentCorpus(TextAlignmentCorpus):
    @overload
    def __init__(self, *args: TextAlignmentCollection) -> None:
        ...

    @overload
    def __init__(self, text_alignment_collections: Iterable[TextAlignmentCollection]) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        text_alignment_collections: Iterable[TextAlignmentCollection] = (
            args if len(args) == 0 or isinstance(args[0], TextAlignmentCollection) else args[0]
        )
        self._text_alignment_collections = {tac.id: tac for tac in text_alignment_collections}

    @property
    def text_alignment_collections(self) -> Iterable[TextAlignmentCollection]:
        return sorted(self._text_alignment_collections.values(), key=lambda tac: tac.sort_key)

    def __getitem__(self, id: str) -> Optional[TextAlignmentCollection]:
        return self._text_alignment_collections.get(id)

    def _add_text_alignment_collection(self, alignments: TextAlignmentCollection) -> None:
        self._text_alignment_collections[alignments.id] = alignments

    def _get_rows(self) -> Generator[TextAlignmentCorpusRow, None, None]:
        for tac in self.text_alignment_collections:
            with tac.get_rows() as rows:
                yield from rows
