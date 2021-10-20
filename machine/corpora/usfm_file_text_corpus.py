from pathlib import Path
from typing import List, Optional

from ..scripture.verse_ref import Versification, VersificationType
from ..tokenization.tokenizer import Tokenizer
from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_file_text import UsfmFileText
from .usfm_stylesheet import UsfmStylesheet


class UsfmFileTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        word_tokenizer: Tokenizer[str, int, str],
        stylesheet_filename: StrPath,
        encoding: str,
        project_dir: StrPath,
        versification: Optional[Versification] = None,
        include_markers: bool = False,
    ) -> None:
        if versification is None:
            versification = Versification.get_builtin(VersificationType.ENGLISH)
        self._versification = versification
        stylesheet = UsfmStylesheet(stylesheet_filename)
        texts: List[UsfmFileText] = []
        for sfm_filename in Path(project_dir).glob("*.SFM"):
            texts.append(
                UsfmFileText(word_tokenizer, stylesheet, encoding, sfm_filename, versification, include_markers)
            )
        super().__init__(texts)

    @property
    def versification(self) -> Versification:
        return self._versification
