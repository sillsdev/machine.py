import os
import xml.etree.ElementTree as etree
from io import TextIOWrapper
from typing import List
from zipfile import ZipFile

from ..scripture.verse_ref import Versification, VersificationType
from ..utils.typeshed import StrPath
from .scripture_text_corpus import ScriptureTextCorpus
from .usx_zip_text import UsxZipText


class DblBundleTextCorpus(ScriptureTextCorpus):
    _SUPPORTED_VERSIONS = {"2.0", "2.1", "2.2"}

    def __init__(self, filename: StrPath) -> None:
        with ZipFile(filename, "r") as archive:
            with archive.open("metadata.xml", "r") as stream:
                doc = etree.parse(stream)
            version = doc.getroot().get("version", "2.0")
            parts = version.split(".", maxsplit=3)
            if f"{parts[0]}.{parts[1]}" not in DblBundleTextCorpus._SUPPORTED_VERSIONS:
                raise RuntimeError("Unsupported version of DBL bundle.")

            versification_entry = next(
                (zi for zi in archive.filelist if os.path.basename(zi.filename) == "versification.vrs"), None
            )
            if versification_entry is not None:
                with archive.open(versification_entry, "r") as stream:
                    abbr = doc.getroot().findtext("./identification/abbreviation", "")
                    versification = Versification.parse(
                        TextIOWrapper(stream, encoding="utf-8-sig"), "versification.vrs", fallback_name=abbr
                    )
            else:
                versification = Versification.get_builtin(VersificationType.ENGLISH)

        texts: List[UsxZipText] = []
        for content_elem in doc.getroot().findall("./publications/publication[@default='true']/structure/content"):
            texts.append(UsxZipText(content_elem.get("role", ""), filename, content_elem.get("src", ""), versification))
        super().__init__(versification, texts)
