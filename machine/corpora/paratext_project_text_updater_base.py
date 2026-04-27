from abc import ABC
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from .paratext_project_file_handler import ParatextProjectFileHandler
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .update_usfm_parser_handler import (
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
)
from .usfm_parser import UsfmParser
from .usfm_token import UsfmTokenType
from .usfm_tokenizer import UsfmToken, UsfmTokenizer
from .usfm_update_block_handler import UsfmUpdateBlockHandler, UsfmUpdateBlockHandlerError
from ..utils.string_utils import parse_integer


class ParatextProjectTextUpdaterBase(ABC):
    def __init__(
        self,
        paratext_project_file_handler: ParatextProjectFileHandler,
        settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase],
    ) -> None:
        self._paratext_project_file_handler = paratext_project_file_handler
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def update_usfm(
        self,
        book_id: str,
        rows: Optional[Sequence[UpdateUsfmRow]] = None,
        chapters: Optional[Sequence[int]] = None,
        full_name: Optional[str] = None,
        text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_EXISTING,
        paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
        preserve_paragraph_styles: Optional[Union[Iterable[str], str]] = None,
        update_block_handlers: Optional[Iterable[UsfmUpdateBlockHandler]] = None,
        remarks: Optional[Iterable[Tuple[int, str]]] = None,
        error_handler: Optional[Callable[[UsfmUpdateBlockHandlerError], bool]] = None,
        compare_segments: bool = False,
    ) -> Optional[str]:
        file_name: str = self._settings.get_book_file_name(book_id)
        if not self._paratext_project_file_handler.exists(file_name):
            return None
        with self._paratext_project_file_handler.open(file_name) as sfm_file:
            usfm: str = sfm_file.read().decode(self._settings.encoding)
        handler = UpdateUsfmParserHandler(
            rows,
            None if full_name is None else f"- {full_name}",
            text_behavior,
            paragraph_behavior,
            embed_behavior,
            style_behavior,
            preserve_paragraph_styles,
            update_block_handlers=update_block_handlers,
            remarks=remarks,
            error_handler=error_handler,
            compare_segments=compare_segments,
        )
        try:
            tokenizer = UsfmTokenizer(self._settings.stylesheet)
            tokens = tokenizer.tokenize(usfm)
            tokens = filter_tokens_by_chapter(tokens, chapters)
            parser = UsfmParser(tokens, handler, self._settings.stylesheet, self._settings.versification)
            parser.process_tokens()
            return handler.get_usfm(self._settings.stylesheet)
        except Exception as e:
            error_message = (
                f"An error occurred while parsing the usfm for '{book_id}'"
                f"{f' in project {self._settings.name}' if self._settings.name else ''}"
                f". Error: '{e}'"
            )
            raise RuntimeError(error_message) from e


def filter_tokens_by_chapter(
    tokens: Sequence[UsfmToken], chapters: Optional[Sequence[int]] = None
) -> Sequence[UsfmToken]:
    if chapters is None:
        return tokens
    tokens_within_chapters: List[UsfmToken] = []
    in_chapter: bool = False
    in_id_marker: bool = False
    for index, token in enumerate(tokens):
        if index == 0 and token.marker == "id":
            in_id_marker = True
            if 1 in chapters:
                in_chapter = True
        elif in_id_marker and token.marker is not None and token.marker != "id":
            in_id_marker = False
        elif token.type == UsfmTokenType.CHAPTER:
            chapter_num = parse_integer(token.data) if token.data else None
            if chapter_num is not None and chapter_num in chapters:
                in_chapter = True
            else:
                in_chapter = False

        if in_id_marker or in_chapter:
            tokens_within_chapters.append(token)
    return tokens_within_chapters
