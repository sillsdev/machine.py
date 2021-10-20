import os
import xml.etree.ElementTree as etree
from io import TextIOWrapper
from typing import List
from zipfile import ZipFile

from ..scripture.verse_ref import Versification, VersificationType
from ..tokenization.tokenizer import Tokenizer
from ..utils.typeshed import StrPath
from .dbl_bundle_text import DblBundleText
from .scripture_text_corpus import ScriptureTextCorpus


class DblBundleTextCorpus(ScriptureTextCorpus):
    _SUPPORTED_VERSIONS = {"2.0", "2.1"}

    def __init__(self, word_tokenizer: Tokenizer[str, int, str], filename: StrPath) -> None:
        with ZipFile(filename, "r") as archive:
            with archive.open("metadata.xml", "r") as stream:
                doc = etree.parse(stream)
            if doc.getroot().get("version") not in DblBundleTextCorpus._SUPPORTED_VERSIONS:
                raise RuntimeError("Unsupported version of DBL bundle.")

            versification_entry = next(
                (zi for zi in archive.filelist if os.path.basename(zi.filename) == "versification.vrs"), None
            )
            if versification_entry is not None:
                with archive.open(versification_entry, "r") as stream:
                    abbr = doc.getroot().findtext("./identification/abbreviation", "")
                    self._versification = Versification.parse(
                        TextIOWrapper(stream, encoding="utf-8-sig"), "versification.vrs", fallback_name=abbr
                    )
            else:
                self._versification = Versification.get_builtin(VersificationType.ENGLISH)

        texts: List[DblBundleText] = []
        for content_elem in doc.getroot().findall("./publications/publication[@default='true']/structure/content"):
            texts.append(
                DblBundleText(
                    word_tokenizer,
                    content_elem.get("role", ""),
                    filename,
                    content_elem.get("src", ""),
                    self._versification,
                )
            )
        super().__init__(texts)

    @property
    def versification(self) -> Versification:
        return self._versification
