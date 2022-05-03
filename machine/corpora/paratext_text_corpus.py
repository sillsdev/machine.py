import xml.etree.ElementTree as etree
from pathlib import Path
from typing import List

from ..scripture.verse_ref import Versification
from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .corpora_utils import get_encoding
from .scripture_text_corpus import ScriptureTextCorpus
from .usfm_file_text import UsfmFileText
from .usfm_stylesheet import UsfmStylesheet


class ParatextTextCorpus(ScriptureTextCorpus):
    def __init__(self, project_dir: StrPath, include_markers: bool = False) -> None:
        if not isinstance(project_dir, Path):
            project_dir = Path(project_dir)
        settings_filename = project_dir / "Settings.xml"
        if not settings_filename.is_file():
            settings_filename = next(project_dir.glob("*.ssf"), Path())
        if not settings_filename.is_file():
            raise RuntimeError("The project directory does not contain a settings file.")

        with settings_filename.open("rb") as settings_file:
            settings_tree = etree.parse(settings_file)

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
        custom_versification_filename = project_dir / "custom.vrs"
        if custom_versification_filename.is_file():
            guid = settings_tree.getroot().findtext("Guid", "")
            versification_name = f"{versification.name}-{guid}"
            versification = Versification.load(custom_versification_filename, versification, versification_name)

        stylesheet_name = settings_tree.getroot().findtext("StyleSheet", "usfm.sty")
        stylesheet_filename = project_dir / stylesheet_name
        if not stylesheet_filename.is_file() and stylesheet_name != "usfm_sb.sty":
            stylesheet_filename = project_dir / "usfm.sty"
        custom_stylesheet_filename = project_dir / "custom.sty"
        stylesheet = UsfmStylesheet(
            stylesheet_filename, custom_stylesheet_filename if custom_stylesheet_filename.is_file() else None
        )

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

        texts: List[UsfmFileText] = []
        for sfm_filename in project_dir.glob(f"{prefix}*{suffix}"):
            texts.append(UsfmFileText(stylesheet, encoding, sfm_filename, versification, include_markers))
        super().__init__(versification, texts)
