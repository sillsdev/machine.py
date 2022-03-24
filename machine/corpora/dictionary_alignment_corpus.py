from typing import Iterable, Optional, overload

from .alignment_collection import AlignmentCollection
from .alignment_corpus import AlignmentCorpus


class DictionaryAlignmentCorpus(AlignmentCorpus):
    @overload
    def __init__(self, *alignment_collections: AlignmentCollection) -> None:
        ...

    @overload
    def __init__(self, alignment_collections: Iterable[AlignmentCollection]) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        alignment_collections: Iterable[AlignmentCollection]
        if len(args) == 0:
            alignment_collections = kwargs.get("alignment_collections", [])
        elif isinstance(args[0], AlignmentCollection):
            alignment_collections = args
        else:
            alignment_collections = args[0]
        self._alignment_collections = {ac.id: ac for ac in alignment_collections}

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return sorted(self._alignment_collections.values(), key=lambda ac: ac.sort_key)

    def __getitem__(self, id: str) -> Optional[AlignmentCollection]:
        return self._alignment_collections.get(id)

    def _add_alignment_collection(self, alignments: AlignmentCollection) -> None:
        self._alignment_collections[alignments.id] = alignments
