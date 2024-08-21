from pathlib import Path
from typing import List

from ..utils.typeshed import StrPath
from .file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_file_text import UsfmFileText


class ParatextTextCorpus(ScriptureTextCorpus):
    def __init__(self, project_dir: StrPath, include_markers: bool = False, include_all_text: bool = False) -> None:
        parser = FileParatextProjectSettingsParser(project_dir)
        settings = parser.parse()

        versification = settings.versification

        texts: List[UsfmFileText] = []
        for sfm_filename in Path(project_dir).glob(f"{settings.file_name_prefix}*{settings.file_name_suffix}"):
            book_id = settings.get_book_id(sfm_filename.name)
            if book_id:
                texts.append(
                    UsfmFileText(
                        settings.stylesheet,
                        settings.encoding,
                        book_id,
                        sfm_filename,
                        versification,
                        include_markers,
                        include_all_text,
                        settings.name,
                    )
                )

        super().__init__(versification, texts)
