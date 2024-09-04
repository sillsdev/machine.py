from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Sequence, Tuple, Union

from ..utils.typeshed import StrPath
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .scripture_ref import ScriptureRef
from .update_usfm_parser_handler import UpdateUsfmParserHandler
from .usfm_parser import parse_usfm


class ParatextProjectTextUpdaterBase(ABC):
    def __init__(self, settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase]) -> None:
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def update_usfm(
        self,
        book_id: str,
        rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
        full_name: Optional[str] = None,
        strip_all_text: bool = False,
        prefer_existing_text: bool = True,
    ) -> Optional[str]:
        file_name: str = self._settings.get_book_file_name(book_id)
        if not self._exists(file_name):
            return None
        with self._open(file_name) as sfm_file:
            usfm: str = sfm_file.read().decode(self._settings.encoding)
        handler = UpdateUsfmParserHandler(
            rows, None if full_name is None else f"- {full_name}", strip_all_text, prefer_existing_text
        )
        try:
            parse_usfm(usfm, handler, self._settings.stylesheet, self._settings.versification)
            return handler.get_usfm(self._settings.stylesheet)
        except Exception as e:
            error_message = (
                f"An error occurred while parsing the usfm for '{book_id}'"
                f"{f' in project {self._settings.name}' if self._settings.name else ''}"
                f". Error: '{e}'"
            )
            raise RuntimeError(error_message) from e

    @abstractmethod
    def _exists(self, file_name: StrPath) -> bool: ...

    @abstractmethod
    def _open(self, file_name: StrPath) -> BinaryIO: ...
