from typing import Generator, Iterable, Optional, Tuple, cast

from ..scripture import ORIGINAL_VERSIFICATION
from ..scripture.canon import book_id_to_number, book_number_to_id, is_canonical
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
    def __init__(self, book_num: int, versification: Versification) -> None:
        super().__init__(book_number_to_id(book_num), versification)

    def _get_rows(self) -> Generator[TextRow, None, None]:
        b = book_id_to_number(self.id)
        for c in range(1, self.versification.get_last_chapter(b) + 1):
            for v in range(1, self.versification.get_last_verse(b, c) + 1):
                vref = self._create_verse_ref(str(c), str(v))
                if not self._versification.is_excluded(vref.bbbcccvvv):
                    yield from self._create_rows(vref)


def create_versification_ref_corpus(versification: Versification = ORIGINAL_VERSIFICATION) -> ScriptureTextCorpus:
    return ScriptureTextCorpus(
        versification,
        (
            _VersificationRefCorpusText(b, versification)
            for b in range(1, versification.get_last_book() + 1)
            if is_canonical(b)
            and (
                versification.get_last_chapter(b) != 1
                or versification.get_last_verse(b, versification.get_last_chapter(b)) != 1
            )
            and (b < 87 or b > 92)
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
            cur_trg_line = ""
            cur_trg_line_range = True
            for row in rows:
                ref: VerseRef = row.ref
                if cur_ref is not None and ref.compare_to(cur_ref, compare_segments=False) != 0:
                    yield "<range>" if cur_trg_line_range else cur_trg_line, cur_ref, cur_trg_ref
                    cur_trg_line_range = cur_trg_line_range or len(cur_trg_line) > 0
                    cur_trg_line = ""
                    cur_trg_ref = None

                cur_ref = ref
                if cur_trg_ref is None and len(row.target_refs) > 0:
                    cur_trg_ref = cast(VerseRef, row.target_refs[0])
                elif cur_trg_ref is not None and len(row.target_refs) > 0 and cur_trg_ref != row.target_refs[0]:
                    cur_trg_ref = cur_trg_ref.copy()
                    cur_trg_ref.simplify()
                    trg_ref = cast(VerseRef, row.target_refs[0])
                    if cur_trg_ref < row.target_refs[0]:
                        start_ref = cur_trg_ref
                        end_ref = trg_ref
                    else:
                        start_ref = trg_ref
                        end_ref = cur_trg_ref
                    if start_ref.chapter == end_ref.chapter:
                        if start_ref.verse_num != end_ref.verse_num:
                            cur_trg_ref = VerseRef.from_range(start_ref, end_ref)
                    else:
                        cur_trg_ref = end_ref
                if not row.is_target_in_range or row.is_target_range_start or len(row.target_text) > 0:
                    if len(row.target_text) > 0:
                        if len(cur_trg_line) > 0:
                            cur_trg_line += " "
                        cur_trg_line += row.target_text
                    cur_trg_line_range = False
            if cur_ref is not None:
                yield "<range>" if cur_trg_line_range else cur_trg_line, cur_ref, cur_trg_ref

    return ContextManagedGenerator(extract())
