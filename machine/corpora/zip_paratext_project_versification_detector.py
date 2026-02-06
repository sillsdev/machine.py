from zipfile import ZipFile

from .paratext_project_versification_error_detector_base import ParatextProjectVersificationErrorDetectorBase
from .zip_paratext_project_file_handler import ZipParatextProjectFileHandler
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ZipParatextProjectVersificationErrorDetector(ParatextProjectVersificationErrorDetectorBase):
    def __init__(self, archive: ZipFile):
        super().__init__(ZipParatextProjectFileHandler(archive), ZipParatextProjectSettingsParser(archive).parse())
