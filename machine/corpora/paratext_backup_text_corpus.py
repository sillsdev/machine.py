from typing import List
from zipfile import ZipFile

import regex as re

from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_zip_text import UsfmZipText
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ParatextBackupTextCorpus(ScriptureTextCorpus):
    def __init__(self, filename: StrPath, include_markers: bool = False, include_all_text: bool = False) -> None:
        with ZipFile(filename, "r") as archive:
            parser = ZipParatextProjectSettingsParser(archive)
            settings = parser.parse()

            versification = settings.versification
            regex = re.compile(f"^{re.escape(settings.file_name_prefix)}.*{re.escape(settings.file_name_suffix)}$")

            texts: List[UsfmZipText] = []
            for sfm_entry in (zi for zi in archive.filelist if regex.match(zi.filename)):
                texts.append(
                    UsfmZipText(
                        settings.stylesheet,
                        settings.encoding,
                        filename,
                        sfm_entry.filename,
                        versification,
                        include_markers,
                        include_all_text,
                    )
                )

        super().__init__(versification, texts)
