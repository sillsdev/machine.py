from typing import List, Optional, Sequence, Tuple

from machine.corpora.scripture_update_block_handler_first_elements_first import (
    ScriptureUpdateBlockHandlerFirstElementsFirst,
)

from machine.corpora.scripture_update_block_handler_base import ScriptureUpdateBlockHandlerBase
from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import (
    FileParatextProjectTextUpdater,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    parse_usfm,
)


def test_preserve_paragraphs():
    rows = [
        (scr_ref("MAT 1:1"), str("U1")),
        (
            scr_ref("MAT 1:1/1:f"),
            str("UF1"),
        ),
        (scr_ref("MAT 1:2"), str("U2")),
        (
            scr_ref("MAT 1:2/1:f"),
            str("UF2"),
        ),
        (scr_ref("MAT 1:3"), str("U3")),
        (
            scr_ref("MAT 1:3/1:f"),
            str("UF3"),
        ),
    ]
    usfm = r"""\id MAT
\c 1
\v 1 \f \ft \fm ' \fm* hello world \f* it comes first
\v 2 it comes \f \ft hello \fm ' \fm* world \f* middling
\v 3 it comes last \f \ft  hello world \fm ' \fm* \f* 
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT
\c 1
\v 1 U1 \f \ft UF1 \fm ' \fm*\f* 
\v 2 U2 \f \ft UF2 \fm ' \fm*\f* 
\v 3 U3 \f \ft UF3 \fm ' \fm*\f* 
"""

    assess(target, result)

    target_first_element = update_usfm(
        rows, usfm, update_block_handlers=[ScriptureUpdateBlockHandlerFirstElementsFirst()]
    )
    result_first_element = r"""\id MAT
\c 1
\v 1 \f \ft \fm ' \fm* UF1 \f* U1 
\v 2 U2 \f \ft UF2 \fm ' \fm*\f* 
\v 3 U3 \f \ft UF3 \fm ' \fm*\f* 
"""
    assess(target_first_element, result_first_element)


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def update_usfm(
    rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
    source: Optional[str] = None,
    id_text: Optional[str] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_NEW,
    paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
    preserve_paragraph_styles: Optional[Sequence[str]] = None,
    update_block_handlers: Optional[list[ScriptureUpdateBlockHandlerBase]] = None,
) -> Optional[str]:
    if source is None:
        updater = FileParatextProjectTextUpdater(USFM_TEST_PROJECT_PATH)
        return updater.update_usfm(
            "MAT",
            rows,
            id_text,
            text_behavior,
            paragraph_behavior,
            embed_behavior,
            style_behavior,
            preserve_paragraph_styles,
            update_block_handlers,
        )
    else:
        source = source.strip().replace("\r\n", "\n") + "\r\n"
        updater = UpdateUsfmParserHandler(
            rows,
            id_text,
            text_behavior,
            paragraph_behavior,
            embed_behavior,
            style_behavior,
            preserve_paragraph_styles,
            update_block_handlers,
        )
        parse_usfm(source, updater)
        return updater.get_usfm()


def assess(target: Optional[str], truth: str) -> None:
    assert target is not None
    for target_line, truth_line in zip(target.split("\n"), truth.split("\n")):
        assert target_line.strip() == truth_line.strip()


def read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
