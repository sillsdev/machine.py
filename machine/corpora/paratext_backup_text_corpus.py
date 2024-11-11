from typing import List
from zipfile import ZipFile

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

            texts: List[UsfmZipText] = []
            for sfm_entry in archive.filelist:
                book_id = settings.get_book_id(sfm_entry.filename)
                if book_id:
                    text = UsfmZipText(
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
                    with text.get_rows() as rows:
                        row = next(rows, None)
                        if row and row.ref.book != book_id:
                            if row.ref.book == "":
                                raise ValueError(f"The \\id tag in {sfm_entry.filename} is invalid.")
                            raise ValueError(
                                f"The \\id tag {row.ref.book} in {sfm_entry.filename}"
                                f" does not match filename book id {book_id}."
                            )
                    texts.append(text)

        super().__init__(versification, texts)
