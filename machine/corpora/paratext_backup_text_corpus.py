import xml.etree.ElementTree as etree
from io import TextIOWrapper
from tempfile import TemporaryFile
from typing import List, Optional
from zipfile import ZipFile, ZipInfo

import regex as re

from ..scripture.verse_ref import Versification
from ..utils.file_utils import detect_encoding_from_stream
from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .corpora_utils import find_entry, get_encoding, get_entry
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_stylesheet import UsfmStylesheet
from .usfm_zip_text import UsfmZipText


class ParatextBackupTextCorpus(ScriptureTextCorpus):
    def __init__(self, filename: StrPath, include_markers: bool = False) -> None:
        with ZipFile(filename, "r") as archive:
            settings_entry = get_entry(archive, "Settings.xml")
            if settings_entry is None:
                settings_entry = find_entry(archive, lambda zi: zi.filename.endswith(".ssf"))
            if settings_entry is None:
                raise ValueError("The project backup does not contain a settings file.")

            with archive.open(settings_entry, "r") as file:
                settings_tree = etree.parse(file)

            encoding_str = settings_tree.getroot().findtext("Encoding", "65001")
            code_page = parse_integer(encoding_str)
            if code_page is None:
                raise NotImplementedError(
                    f"The project uses a legacy encoding that requires TECKit, map file: {encoding_str}."
                )
            encoding = get_encoding(code_page)
            if encoding is None:
                raise RuntimeError(f"Code page {code_page} not supported.")

            versification_type = int(settings_tree.getroot().findtext("Versification", "4"))
            versification = Versification.get_builtin(versification_type)
            custom_versification_entry = get_entry(archive, "custom.vrs")
            if custom_versification_entry is not None:
                guid = settings_tree.getroot().findtext("Guid", "")
                versification_name = f"{versification.name}-{guid}"
                try:
                    versification = _load_versification_from_entry(
                        archive,
                        custom_versification_entry,
                        "custom.vrs",
                        versification,
                        versification_name,
                        encoding="utf-8-sig",
                    )
                except UnicodeDecodeError:
                    with archive.open(custom_versification_entry, "r") as file:
                        vers_encoding = detect_encoding_from_stream(file)
                    versification = _load_versification_from_entry(
                        archive,
                        custom_versification_entry,
                        "custom.vrs",
                        versification,
                        versification_name,
                        vers_encoding,
                    )

            stylesheet_name = settings_tree.getroot().findtext("StyleSheet", "usfm.sty")
            stylesheet_entry = get_entry(archive, stylesheet_name)
            if stylesheet_entry is None and stylesheet_name != "usfm_sb.sty":
                stylesheet_entry = get_entry(archive, "usfm.sty")
            custom_stylesheet_entry = get_entry(archive, "custom.sty")
            with TemporaryFile() as stylesheet_temp_file, TemporaryFile() as custom_stylesheet_temp_file:
                stylesheet_path = "usfm.sty"
                if stylesheet_entry is not None:
                    with archive.open(stylesheet_entry, "r") as file:
                        stylesheet_temp_file.write(file.read())
                    stylesheet_path = stylesheet_temp_file.name
                stylesheet_temp_file.close()
                custom_stylesheet_path: Optional[str] = None
                if custom_stylesheet_entry is not None:
                    with archive.open(custom_stylesheet_entry, "r") as file:
                        custom_stylesheet_temp_file.write(file.read())
                    custom_stylesheet_path = custom_stylesheet_temp_file.name
                custom_stylesheet_temp_file.close()
                stylesheet = UsfmStylesheet(stylesheet_path, custom_stylesheet_path)

            prefix = ""
            suffix = ".SFM"
            naming_elem = settings_tree.getroot().find("Naming")
            if naming_elem is not None:
                pre_part = naming_elem.get("PrePart", "")
                if pre_part != "":
                    prefix = pre_part
                post_part = naming_elem.get("PostPart", "")
                if post_part != "":
                    suffix = post_part

            regex = re.compile(f"^{re.escape(prefix)}.*{re.escape(suffix)}$")
            texts: List[UsfmZipText] = []
            for sfm_entry in (zi for zi in archive.filelist if regex.match(zi.filename)):
                texts.append(
                    UsfmZipText(stylesheet, encoding, filename, sfm_entry.filename, versification, include_markers)
                )

        super().__init__(versification, texts)


def _load_versification_from_entry(
    archive: ZipFile,
    entry: ZipInfo,
    filename: StrPath,
    base_versification: Versification,
    fallback_name: str,
    encoding: str,
) -> Versification:
    with archive.open(entry, "r") as file:
        stream = TextIOWrapper(file, encoding=encoding)
        return Versification.parse(
            stream, filename, Versification(fallback_name, filename, base_versification), fallback_name
        )
