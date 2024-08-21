from pathlib import Path
from typing import List, Optional

from ..scripture import ENGLISH_VERSIFICATION
from ..scripture.verse_ref import Versification
from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_file_text import UsfmFileText
from .usfm_stylesheet import UsfmStylesheet


class UsfmFileTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        project_dir: StrPath,
        stylesheet_filename: StrPath = "usfm.sty",
        encoding: str = "utf-8-sig",
        versification: Optional[Versification] = None,
        include_markers: bool = False,
        file_pattern: str = "*.SFM",
        include_all_text: bool = False,
    ) -> None:
        if versification is None:
            versification = ENGLISH_VERSIFICATION
        stylesheet = UsfmStylesheet(stylesheet_filename)
        texts: List[UsfmFileText] = []
        for sfm_filename in Path(project_dir).glob(file_pattern):
            id = _get_id(sfm_filename, encoding)
            if id:
                texts.append(
                    UsfmFileText(
                        stylesheet, encoding, id, sfm_filename, versification, include_markers, include_all_text
                    )
                )
        super().__init__(versification, texts)


def _get_id(filename: StrPath, encoding: str) -> Optional[str]:
    with open(filename, "r", encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line.startswith("\\id "):
                id = line[4:]
                index = id.find(" ")
                if index != -1:
                    id = id[:index]
                return id.strip().upper()
    return None
