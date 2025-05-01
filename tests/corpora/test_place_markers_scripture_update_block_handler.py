from typing import List, Optional, Sequence, Tuple

from machine.corpora import (
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    parse_usfm,
)
from machine.corpora.place_markers_scripture_update_block_handler import PlaceMarkersScriptureUpdateBlockHandler
from machine.corpora.scripture_update_block_handler import ScriptureUpdateBlockHandler
from machine.jobs.translation_file_service import PretranslationInfo
from machine.tokenization import LatinWordTokenizer

TOKENIZER = LatinWordTokenizer()


def test_paragraph_markers():
    source = "This is the first paragraph. This text is in English, and this test is for paragraph markers."
    pretranslation = "Este es el primer párrafo. Este texto está en inglés y esta prueba es para marcadores de párrafo."
    rows = [(scr_ref("MAT 1:1"), str(pretranslation))]
    usfm = r"""\id MAT
\c 1
\v 1 This is the first paragraph.
\p This text is in English,
\p and this test is for paragraph markers.
\p
"""

    pt_info = [
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:1"],
            translation=pretranslation,
            source_toks=[t for t in TOKENIZER.tokenize(source)],
            translation_toks=[t for t in TOKENIZER.tokenize(pretranslation)],
            alignment="0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 10-10 12-11 13-12 14-13 15-14 16-15 17-18 18-16 19-19",
        ),
    ]
    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 Este es el primer párrafo.
\p Este texto está en inglés
\p y esta prueba es para marcadores de párrafo.
\p
"""
    assess(target, result)

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 Este es el primer párrafo. Este texto está en inglés y esta prueba es para marcadores de párrafo.
"""
    assess(target, result)


def test_style_markers():
    source = "This is the first sentence. This text is in English, and this test is for style markers."
    pretranslation = "Esta es la primera oración. Este texto está en inglés y esta prueba es para marcadores de estilo."
    rows = [(scr_ref("MAT 1:1"), str(pretranslation))]
    usfm = r"""\id MAT
\c 1
\v 1 This is the \w first\w* sentence. This text is in \w English\w*, and this test is \w for\w* style markers.
"""

    pt_info = [
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:1"],
            translation=pretranslation,
            source_toks=[t for t in TOKENIZER.tokenize(source)],
            translation_toks=[t for t in TOKENIZER.tokenize(pretranslation)],
            alignment="0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 10-10 12-11 13-12 14-13 15-14 16-15 17-18 18-16 19-19",
        ),
    ]
    target = update_usfm(
        rows,
        usfm,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 Esta es la \w primera \w*oración. Este texto está en \w inglés \w*y esta prueba es \w para \w*marcadores de estilo.
"""
    # TODO: the spacing before/after end markers is incorrect,
    # but this is an issue with how the is USFM is generated from the tokens
    assess(target, result)

    target = update_usfm(
        rows,
        usfm,
        style_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 Esta es la primera oración. Este texto está en inglés y esta prueba es para marcadores de estilo.
"""
    assess(target, result)


def test_embeds():
    rows = [
        (scr_ref("MAT 1:1"), str("New verse 1")),
        (scr_ref("MAT 1:2"), str("New verse 2")),
        (scr_ref("MAT 1:3"), str("New verse 3")),
        (scr_ref("MAT 1:4"), str("New verse 4")),
        (scr_ref("MAT 1:4/1:f"), str("New embed text")),
        (scr_ref("MAT 1:5"), str("New verse 5")),
        (scr_ref("MAT 1:6"), str("New verse 6")),
        (scr_ref("MAT 1:6/1:f"), str("New verse 6 embed text")),
    ]
    usfm = r"""\id MAT
\c 1
\v 1 \f \fr 1.1 \ft Some note \f*Start of sentence embed
\v 2 Middle of sentence \f \fr 1.2 \ft Some other note \f*embed
\v 3 End of sentence embed\f \fr 1.3 \ft A third note \f*
\v 4 Updated embed\f \fr 1.4 \ft A fourth note \f*
\v 5 Embed with style markers \f \fr 1.5 \ft A \+w stylish\+w* note \f*
\v 6 Updated embed with style markers \f \fr 1.6 \ft Another \+w stylish\+w* note \f*
"""

    pt_info = [
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:1"],
            translation="New verse 1",
            source_toks=["Start", "of", "sentence", "embed"],
            translation_toks=["New", "verse", "1"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:2"],
            translation="New verse 2",
            source_toks=["Middle", "of", "sentence", "embed"],
            translation_toks=["New", "verse", "2"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:3"],
            translation="New verse 3",
            source_toks=["End", "of", "sentence", "embed"],
            translation_toks=["New", "verse", "3"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:4"],
            translation="New verse 4",
            source_toks=["Updated", "embed"],
            translation_toks=["New", "verse", "4"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:4/1:f"],
            translation="New embed text",
            source_toks=["A", "fourth", "note"],
            translation_toks=["New", "embed", "text"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:5"],
            translation="New verse 5",
            source_toks=["Embed", "with", "style", "markers"],
            translation_toks=["New", "verse", "5"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:6"],
            translation="New verse 6",
            source_toks=["Updated", "embed", "with", "style", "markers"],
            translation_toks=["New", "verse", "6"],
            alignment="",
        ),
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:6/1:f"],
            translation="New verse 6 embed text",
            source_toks=["Another", "stylish", "note"],
            translation_toks=["New", "verse", "6", "embed", "text"],
            alignment="",
        ),
    ]
    target = update_usfm(
        rows,
        usfm,
        embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 New verse 1 \f \fr 1.1 \ft Some note \f*
\v 2 New verse 2 \f \fr 1.2 \ft Some other note \f*
\v 3 New verse 3 \f \fr 1.3 \ft A third note \f*
\v 4 New verse 4 \f \fr 1.4 \ft New embed text \f*
\v 5 New verse 5 \f \fr 1.5 \ft A \+w stylish\+w* note \f*
\v 6 New verse 6 \f \fr 1.6 \ft New verse 6 embed text \f*
"""
    assess(target, result)

    target = update_usfm(
        rows,
        usfm,
        embed_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 New verse 1
\v 2 New verse 2
\v 3 New verse 3
\v 4 New verse 4
\v 5 New verse 5
\v 6 New verse 6
"""
    assess(target, result)


def test_no_alignment():
    rows = [(scr_ref("MAT 1:1"), str("New paragraph 1 New paragraph 2"))]
    usfm = r"""\id MAT
\c 1
\v 1 Old paragraph 1
\p Old paragraph 2
"""

    pt_info = [
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:1"],
            translation="New paragraph 1 New paragraph 2",
            source_toks=["Old", "paragraph", "1", "Old", "paragraph", "2"],
            translation_toks=["New", "paragraph", "1", "New", "paragraph", "2"],
            alignment="",
        ),
    ]
    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 New paragraph 1 New paragraph 2
\p
"""
    assess(target, result)


def test_changed_text():
    rows = [(scr_ref("MAT 1:1"), str("New paragraph 1 New paragraph 2"))]
    usfm = r"""\id MAT
\c 1
\v 1 Old paragraph 1
\p Old paragraph 2
"""

    pt_info = [
        PretranslationInfo(
            corpusId="",
            textId="",
            refs=["MAT 1:1"],
            translation="Changed paragraph 1 Changed paragraph 2",
            source_toks=["Old", "paragraph", "1", "Old", "paragraph", "2"],
            translation_toks=["Changed", "paragraph", "1", "Changed", "paragraph", "2"],
            alignment="0-0 1-1 2-2 3-3 4-4 5-5",
        ),
    ]
    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersScriptureUpdateBlockHandler(pt_info)],
    )
    result = r"""\id MAT
\c 1
\v 1 New paragraph 1 New paragraph 2
\p
"""
    assess(target, result)


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def update_usfm(
    rows: Sequence[Tuple[Sequence[ScriptureRef], str]],
    source: str,
    id_text: Optional[str] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_NEW,
    paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
    preserve_paragraph_styles: Optional[Sequence[str]] = None,
    update_block_handlers: Optional[list[ScriptureUpdateBlockHandler]] = None,
) -> Optional[str]:
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
        print(truth_line)
        print(target_line)
    for target_line, truth_line in zip(target.split("\n"), truth.split("\n")):
        assert target_line.strip() == truth_line.strip()
