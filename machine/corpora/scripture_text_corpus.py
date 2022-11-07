from typing import Generator, Iterable, List, Optional, Tuple

from ..scripture import ORIGINAL_VERSIFICATION
from ..scripture.canon import book_number_to_id
from ..scripture.verse_ref import VerseRef, Versification
from ..utils.context_managed_generator import ContextManagedGenerator
from .dictionary_text_corpus import DictionaryTextCorpus
from .scripture_text import ScriptureText
from .text_corpus import TextCorpus
from .text_row import TextRow


class ScriptureTextCorpus(DictionaryTextCorpus):
    def __init__(self, versification: Versification, texts: Iterable[ScriptureText] = []) -> None:
        super().__init__(texts)
        self._versification = versification

    @property
    def versification(self) -> Versification:
        return self._versification


class _VersificationRefCorpusText(ScriptureText):
    def __init__(self, book_num: int, versification: Versification, verses_in_chapter: List[int]) -> None:
        super().__init__(book_number_to_id(book_num), versification)
        self._verses_in_chapter = verses_in_chapter

    def _get_rows(self) -> Generator[TextRow, None, None]:
        for c, num_verses in enumerate(self._verses_in_chapter):
            for v in range(num_verses):
                vref = self._create_verse_ref(str(c + 1), str(v + 1))
                if vref.bbbcccvvv not in self._versification.excluded_verses:
                    yield from self._create_rows(vref, "")


def create_versification_ref_corpus(versification: Versification = ORIGINAL_VERSIFICATION) -> ScriptureTextCorpus:
    return ScriptureTextCorpus(
        versification,
        (
            _VersificationRefCorpusText(b + 1, versification, verses_in_chapter)
            for b, verses_in_chapter in enumerate(versification.book_list)
            if verses_in_chapter != [1] and (b < 86 or b > 91)
        ),
    )


def extract_scripture_corpus(
    corpus: TextCorpus,
    ref_corpus: TextCorpus = create_versification_ref_corpus(),
) -> ContextManagedGenerator[Tuple[str, VerseRef, Optional[VerseRef]], None, None]:
    parallel_corpus = ref_corpus.align_rows(corpus, all_source_rows=True)

    def extract() -> Generator[Tuple[str, VerseRef, Optional[VerseRef]], None, None]:
        with parallel_corpus.get_rows() as rows:
            cur_ref: Optional[VerseRef] = None
            cur_trg_ref: Optional[VerseRef] = None
            cur_target_line = ""
            cur_target_line_range = True
            for row in rows:
                ref: VerseRef = row.ref
                if cur_ref is not None and ref.compare_to(cur_ref, compare_segments=False) != 0:
                    yield "<range>" if cur_target_line_range else cur_target_line, cur_ref, cur_trg_ref
                    cur_target_line = ""
                    cur_target_line_range = True
                    cur_trg_ref = None

                cur_ref = ref
                if cur_trg_ref is None and len(row.target_refs) > 0:
                    cur_trg_ref = row.target_refs[0]
                elif cur_trg_ref is not None and len(row.target_refs) > 0 and cur_trg_ref != row.target_refs[0]:
                    cur_trg_ref.simplify()
                    if cur_trg_ref < row.target_refs[0]:
                        start_ref = cur_trg_ref
                        end_ref = row.target_refs[0]
                    else:
                        start_ref = row.target_refs[0]
                        end_ref = cur_trg_ref
                    if start_ref.chapter == end_ref.chapter:
                        cur_trg_ref = VerseRef.from_range(start_ref, end_ref)
                    else:
                        cur_trg_ref = end_ref
                if not row.is_target_in_range or row.is_target_range_start or len(row.target_text) > 0:
                    if len(row.target_text) > 0:
                        if len(cur_target_line) > 0:
                            cur_target_line += " "
                        cur_target_line += row.target_text
                    cur_target_line_range = False
            if cur_ref is not None:
                yield "<range>" if cur_target_line_range else cur_target_line, cur_ref, cur_trg_ref

    return ContextManagedGenerator(extract())
