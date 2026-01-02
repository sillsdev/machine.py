from ..utils.typeshed import StrPath
from .file_paratext_project_file_handler import FileParatextProjectFileHandler
from .file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from .paratext_project_terms_parser_base import ParatextProjectTermsParserBase


class FileParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, project_dir: StrPath) -> None:
        super().__init__(
            FileParatextProjectFileHandler(project_dir), FileParatextProjectSettingsParser(project_dir).parse()
        )
