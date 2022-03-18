from pathlib import Path
from typing import List, Optional

from ..scripture.verse_ref import Versification, VersificationType
from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_file_text import UsfmFileText
from .usfm_stylesheet import UsfmStylesheet


class UsfmFileTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        stylesheet_filename: StrPath,
        encoding: str,
        project_dir: StrPath,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
        file_pattern: str = "*.SFM",
    ) -> None:
        if versification is None:
            versification = Versification.get_builtin(VersificationType.ENGLISH)
        stylesheet = UsfmStylesheet(stylesheet_filename)
        texts: List[UsfmFileText] = []
        for sfm_filename in Path(project_dir).glob(file_pattern):
            texts.append(UsfmFileText(stylesheet, encoding, sfm_filename, versification, include_markers))
        super().__init__(versification, texts)
