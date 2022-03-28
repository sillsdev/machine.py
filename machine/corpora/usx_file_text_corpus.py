from pathlib import Path
from typing import List, Optional

from ..scripture.verse_ref import Versification
from ..utils.typeshed import StrPath
from .corpora_utils import get_usx_versification
from .scripture_text_corpus import ScriptureTextCorpus
from .usx_file_text import UsxFileText


class UsxFileTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        project_dir: StrPath,
        versification: Optional[Versification] = None,
    ) -> None:
        project_dir = Path(project_dir)
        versification = get_usx_versification(project_dir, versification)
        texts: List[UsxFileText] = []
        for filename in project_dir.glob("*.usx"):
            texts.append(UsxFileText(filename, versification))
        super().__init__(versification, texts)
