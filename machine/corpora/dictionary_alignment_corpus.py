from typing import Any, Generator, Iterable, Optional, overload

from .alignment_collection import AlignmentCollection
from .alignment_corpus import AlignmentCorpus
from .alignment_row import AlignmentRow


class DictionaryAlignmentCorpus(AlignmentCorpus):
    @overload
    def __init__(self, *args: AlignmentCollection) -> None:
        ...

    @overload
    def __init__(self, text_alignment_collections: Iterable[AlignmentCollection]) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        text_alignment_collections: Iterable[AlignmentCollection] = (
            args if len(args) == 0 or isinstance(args[0], AlignmentCollection) else args[0]
        )
        self._text_alignment_collections = {tac.id: tac for tac in text_alignment_collections}

    @property
    def text_alignment_collections(self) -> Iterable[AlignmentCollection]:
        return sorted(self._text_alignment_collections.values(), key=lambda tac: tac.sort_key)

    def __getitem__(self, id: str) -> Optional[AlignmentCollection]:
        return self._text_alignment_collections.get(id)

    def _add_text_alignment_collection(self, alignments: AlignmentCollection) -> None:
        self._text_alignment_collections[alignments.id] = alignments

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        for tac in self.text_alignment_collections:
            with tac.get_rows() as rows:
                yield from rows
