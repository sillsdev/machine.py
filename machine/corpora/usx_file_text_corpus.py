from pathlib import Path
from typing import List, Optional

from ..scripture.verse_ref import Versification
from ..tokenization.tokenizer import Tokenizer
from ..utils.typeshed import StrPath
from .corpora_helpers import get_usx_versification
from .scripture_text_corpus import ScriptureTextCorpus
from .usx_file_text import UsxFileText


class UsxFileTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        word_tokenizer: Tokenizer[str, int, str],
        project_dir: StrPath,
        versification: Optional[Versification] = None,
    ) -> None:
        project_dir = Path(project_dir)
        self._versification = get_usx_versification(project_dir, versification)
        texts: List[UsxFileText] = []
        for filename in project_dir.glob("*.usx"):
            texts.append(UsxFileText(word_tokenizer, filename, self._versification))
        super().__init__(texts)

    @property
    def versification(self) -> Versification:
        return self._versification
