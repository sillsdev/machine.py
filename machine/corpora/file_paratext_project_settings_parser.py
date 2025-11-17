from ..utils.typeshed import StrPath
from .file_paratext_project_file_handler import FileParatextProjectFileHandler
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase


class FileParatextProjectSettingsParser(ParatextProjectSettingsParserBase):
    def __init__(self, project_dir: StrPath) -> None:
        super().__init__(FileParatextProjectFileHandler(project_dir))
