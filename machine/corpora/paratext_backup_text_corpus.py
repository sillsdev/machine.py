from typing import List, Optional
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_zip_text import UsfmZipText
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ParatextBackupTextCorpus(ScriptureTextCorpus):
    def __init__(
        self,
        filename: StrPath,
        include_markers: bool = False,
        include_all_text: bool = False,
        parent_filename: Optional[StrPath] = None,
    ) -> None:

        parent_settings = None
        if parent_filename is not None:
            with ZipFile(parent_filename, "r") as parent_archive:
                parent_parser = ZipParatextProjectSettingsParser(parent_archive)
                parent_settings = parent_parser.parse()

        with ZipFile(filename, "r") as archive:
            parser = ZipParatextProjectSettingsParser(archive, parent_settings)
            settings = parser.parse()

            versification = settings.versification

            texts: List[UsfmZipText] = []
            for sfm_entry in archive.filelist:
                book_id = settings.get_book_id(sfm_entry.filename)
                if book_id:
                    texts.append(
                        UsfmZipText(
                            settings.stylesheet,
                            settings.encoding,
                            book_id,
                            filename,
                            sfm_entry.filename,
                            versification,
                            include_markers,
                            include_all_text,
                            settings.name,
                        )
                    )

        super().__init__(versification, texts)
