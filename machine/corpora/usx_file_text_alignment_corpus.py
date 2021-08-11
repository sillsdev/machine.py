from pathlib import Path
from typing import Dict, List, Optional

from ..scripture.verse_ref import Versification
from ..tokenization.range_tokenizer import RangeTokenizer
from ..utils.typeshed import StrPath
from .corpora_helpers import get_usx_id, get_usx_versification
from .dictionary_text_alignment_corpus import DictionaryTextAlignmentCorpus
from .usx_file_text_alignment_collection import UsxFileTextAlignmentCollection


class UsxFileTextAlignmentCorpus(DictionaryTextAlignmentCorpus):
    def __init__(
        self,
        src_word_tokenizer: RangeTokenizer[str, int, str],
        trg_word_tokenizer: RangeTokenizer[str, int, str],
        src_project_dir: StrPath,
        trg_project_dir: StrPath,
        src_versification: Optional[Versification] = None,
        trg_versification: Optional[Versification] = None,
    ) -> None:
        src_project_dir = Path(src_project_dir)
        trg_project_dir = Path(trg_project_dir)
        src_versification = get_usx_versification(src_project_dir, src_versification)
        trg_versification = get_usx_versification(trg_project_dir, trg_versification)

        src_filenames = _get_filenames(src_project_dir)
        trg_filenames = _get_filenames(trg_project_dir)

        alignments: List[UsxFileTextAlignmentCollection] = []
        for id in src_filenames.keys() & trg_filenames.keys():
            src_filename = src_filenames[id]
            trg_filename = trg_filenames[id]
            alignments.append(
                UsxFileTextAlignmentCollection(
                    src_word_tokenizer,
                    trg_word_tokenizer,
                    src_filename,
                    trg_filename,
                    src_versification,
                    trg_versification,
                )
            )
        super().__init__(alignments)


def _get_filenames(project_dir: Path) -> Dict[str, Path]:
    return {get_usx_id(f): f for f in project_dir.glob("*.usx")}
