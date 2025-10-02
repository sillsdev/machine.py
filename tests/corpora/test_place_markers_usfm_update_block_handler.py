from typing import List, Optional, Sequence

from machine.corpora import (
    AlignedWordPair,
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
    UsfmUpdateBlockHandler,
    parse_usfm,
)
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

TOKENIZER = LatinWordTokenizer()


def test_paragraph_markers() -> None:
    source = "This is the first paragraph. This text is in English, and this test is for paragraph markers."
    pretranslation = "Este es el primer párrafo. Este texto está en inglés y esta prueba es para marcadores de párrafo."
    align_info = PlaceMarkersAlignmentInfo(
        source_tokens=[t for t in TOKENIZER.tokenize(source)],
        translation_tokens=[t for t in TOKENIZER.tokenize(pretranslation)],
        alignment=to_word_alignment_matrix(
            "0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 10-10 12-11 13-12 14-13 15-14 16-15 17-18 18-16 19-19"
        ),
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.STRIP,
    )
    rows = [UpdateUsfmRow(scr_ref("MAT 1:1"), str(pretranslation), {"alignment_info": align_info})]
    usfm = r"""\id MAT
\c 1
\v 1 This is the first paragraph.
\p This text is in English,
\p and this test is for paragraph markers.
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 Este es el primer párrafo.
\p Este texto está en inglés
\p y esta prueba es para marcadores de párrafo.
"""
    assess(target, result)


def test_style_markers() -> None:
    source = "This is the first sentence. This text is in English, and this test is for style markers."
    pretranslation = "Esta es la primera oración. Este texto está en inglés y esta prueba es para marcadores de estilo."
    align_info = PlaceMarkersAlignmentInfo(
        source_tokens=[t for t in TOKENIZER.tokenize(source)],
        translation_tokens=[t for t in TOKENIZER.tokenize(pretranslation)],
        alignment=to_word_alignment_matrix(
            "0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 10-10 12-11 13-12 14-13 15-14 16-15 17-18 18-16 19-19"
        ),
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
    )
    rows = [UpdateUsfmRow(scr_ref("MAT 1:1"), str(pretranslation), metadata={"alignment_info": align_info})]
    usfm = r"""\id MAT
\c 1
\v 1 This is the \w first\w* sentence. This text is in \w English\w*, and this test is \w for\w* style markers.
"""

    target = update_usfm(
        rows,
        usfm,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 Esta es la \w primera\w* oración. Este texto está en \w inglés\w* y esta prueba es \w para\w* marcadores de estilo.
"""
    assess(target, result)

    align_info = PlaceMarkersAlignmentInfo(
        source_tokens=[t for t in TOKENIZER.tokenize(source)],
        translation_tokens=[t for t in TOKENIZER.tokenize(pretranslation)],
        alignment=to_word_alignment_matrix(
            "0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 10-10 12-11 13-12 14-13 15-14 16-15 17-18 18-16 19-19"
        ),
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.STRIP,
    )
    rows = [UpdateUsfmRow(scr_ref("MAT 1:1"), str(pretranslation), metadata={"alignment_info": align_info})]
    target = update_usfm(
        rows,
        usfm,
        style_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 Esta es la primera oración. Este texto está en inglés y esta prueba es para marcadores de estilo.
"""
    assess(target, result)


# NOTE: Not currently updating embeds, will need to change test when we do
def test_embeds() -> None:
    rows = [
        UpdateUsfmRow(scr_ref("MAT 1:1"), "New verse 1"),
        UpdateUsfmRow(scr_ref("MAT 1:2"), "New verse 2"),
        UpdateUsfmRow(scr_ref("MAT 1:3"), "New verse 3"),
        UpdateUsfmRow(scr_ref("MAT 1:4"), "New verse 4"),
        UpdateUsfmRow(scr_ref("MAT 1:4/1:f"), "New embed text"),
        UpdateUsfmRow(scr_ref("MAT 1:5"), "New verse 5"),
        UpdateUsfmRow(scr_ref("MAT 1:6"), "New verse 6"),
        UpdateUsfmRow(scr_ref("MAT 1:6/1:f"), "New verse 6 embed text"),
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

    target = update_usfm(
        rows,
        usfm,
        embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 New verse 1 \f \fr 1.1 \ft Some note \f*
\v 2 New verse 2 \f \fr 1.2 \ft Some other note \f*
\v 3 New verse 3 \f \fr 1.3 \ft A third note \f*
\v 4 New verse 4 \f \fr 1.4 \ft A fourth note \f*
\v 5 New verse 5 \f \fr 1.5 \ft A \+w stylish\+w* note \f*
\v 6 New verse 6 \f \fr 1.6 \ft Another \+w stylish\+w* note \f*
"""
    assess(target, result)

    target = update_usfm(
        rows,
        usfm,
        embed_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
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


def test_trailing_empty_paragraphs() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "New verse 1",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["Verse", "1"],
                    translation_tokens=["New", "verse", "1"],
                    alignment=to_word_alignment_matrix("0-1 1-2"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 \f embed 1 \f*Verse 1
\p
\b
\q1 \f embed 2 \f*
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 New verse 1 \f embed 1 \f*\f embed 2 \f*
\p
\b
\q1
"""
    assess(target, result)


def test_headers() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "X Y Z",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["A", "B", "C"],
                    translation_tokens=["X", "Y", "Z"],
                    alignment=to_word_alignment_matrix("0-0 1-1 2-2"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:2"),
            "X",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["A"],
                    translation_tokens=["X"],
                    alignment=to_word_alignment_matrix("0-0"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        ),
        UpdateUsfmRow(scr_ref("MAT 1:3"), "Y"),
        UpdateUsfmRow(scr_ref("MAT 1:3/1:s1"), "Updated header"),
    ]
    usfm = r"""\id MAT
\c 1
\s1 Start of chapter header
\p
\v 1 A
\p B
\s1 Mid-verse header
\p C
\s1 Header between verse text and empty end-of-verse paragraphs
\p
\p
\p
\s1 Header after all verse paragraphs
\p
\v 2 A
\s1 Header followed by a reference
\r (reference)
\p
\v 3 B
\s1 Header to be updated
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\s1 Start of chapter header
\p
\v 1 X
\p Y
\s1 Mid-verse header
\p Z
\s1 Header between verse text and empty end-of-verse paragraphs
\p
\p
\p
\s1 Header after all verse paragraphs
\p
\v 2 X
\s1 Header followed by a reference
\r (reference)
\p
\v 3 Y
\s1 Updated header
"""
    assess(target, result)


def test_consecutive_markers() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "New verse 1 WORD",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["Old", "verse", "1", "word"],
                    translation_tokens=["New", "verse", "1", "WORD"],
                    alignment=to_word_alignment_matrix("0-0 1-1 2-2 3-3"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 Old verse 1
\p \qt \+w word\+w*\qt*
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 New verse 1
\p \qt \+w WORD\+w*\qt*
"""
    assess(target, result)


def test_verse_ranges() -> None:
    rows = [
        UpdateUsfmRow(
            [ScriptureRef.parse(f"MAT 1:{i}") for i in range(1, 6)],
            "New verse range text new paragraph 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["Verse", "range", "old", "paragraph", "2"],
                    translation_tokens=["New", "verse", "range", "text", "new", "paragraph", "2"],
                    alignment=to_word_alignment_matrix("0-1 1-2 2-4 3-5 4-6"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1-5 Verse range
\p old paragraph 2
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1-5 New verse range text
\p new paragraph 2
"""
    assess(target, result)


def test_no_update() -> None:
    # Strip paragraphs
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "New paragraph 1 New paragraph 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["Old", "paragraph", "1", "Old", "paragraph", "2"],
                    translation_tokens=["New", "paragraph", "1", "New", "paragraph", "2"],
                    alignment=to_word_alignment_matrix("0-0 1-1 2-2 3-3 4-4 5-5"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 Old paragraph 1
\p Old paragraph 2
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 New paragraph 1 New paragraph 2
"""
    assess(target, result)

    # No alignment
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "New paragraph 1 New paragraph 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=[],
                    translation_tokens=[],
                    alignment=to_word_alignment_matrix(""),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 New paragraph 1 New paragraph 2
\p
"""
    assess(target, result)

    # No text update
    rows = []
    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 Old paragraph 1
\p Old paragraph 2
"""
    assess(target, result)


def test_split_tokens() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "words split words split words split",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["words", "split", "words", "split", "words", "split"],
                    translation_tokens=["words", "split", "words", "split", "words", "split"],
                    alignment=to_word_alignment_matrix("0-0 1-1 2-2 3-3 4-4 5-5"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 words spl
\p it words spl
\p it words split
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 words split
\p words split
\p words split
"""
    assess(target, result)


def test_no_text() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=[],
                    translation_tokens=[],
                    alignment=to_word_alignment_matrix(""),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 \w \w*
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1  \w \w*
"""
    assess(target, result)


def test_consecutive_substring() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "string ring",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["string", "ring"],
                    translation_tokens=["string", "ring"],
                    alignment=to_word_alignment_matrix("0-0 1-1"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 string
\p ring
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 string
\p ring
"""
    assess(target, result)


def test_verses_out_of_order() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "new verse 1 new paragraph 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["verse", "1", "paragraph", "2"],
                    translation_tokens=["new", "verse", "1", "new", "paragraph", "2"],
                    alignment=to_word_alignment_matrix("0-1 1-2 2-4 3-5"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:2"),
            "new verse 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["verse", "2"],
                    translation_tokens=["new", "verse", "2"],
                    alignment=to_word_alignment_matrix("0-1 1-2"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                    style_behavior=UpdateUsfmMarkerBehavior.STRIP,
                )
            },
        ),
    ]
    usfm = r"""\id MAT
\c 1
\v 2 verse 2
\v 1 verse 1
\p paragraph 2
"""

    target = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 2 new verse 2
\v 1 new verse 1
\p new paragraph 2
"""
    assess(target, result)


def test_strip_paragraphs_with_header() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            "new verse 1 new paragraph 2",
            metadata={
                "alignment_info": PlaceMarkersAlignmentInfo(
                    source_tokens=["verse", "1", "paragraph", "2"],
                    translation_tokens=["new", "verse", "1", "new", "paragraph", "2"],
                    alignment=to_word_alignment_matrix("0-1 1-2 2-4 3-5"),
                    paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
                    style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
                )
            },
        )
    ]
    usfm = r"""\id MAT
\c 1
\v 1 verse 1
\s header
\p paragraph 2
\v 2 verse 2
"""

    target = update_usfm(
        rows,
        usfm,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        update_block_handlers=[PlaceMarkersUsfmUpdateBlockHandler()],
    )
    result = r"""\id MAT
\c 1
\v 1 new verse 1 new paragraph 2
\s header
\p
\v 2 verse 2
"""
    assess(target, result)


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def to_word_alignment_matrix(alignment_str: str) -> WordAlignmentMatrix:
    word_pairs = AlignedWordPair.from_string(alignment_str)
    row_count = 0
    column_count = 0
    for pair in word_pairs:
        if pair.source_index + 1 > row_count:
            row_count = pair.source_index + 1
        if pair.target_index + 1 > column_count:
            column_count = pair.target_index + 1
    return WordAlignmentMatrix.from_word_pairs(row_count, column_count, word_pairs)


def update_usfm(
    rows: Sequence[UpdateUsfmRow],
    source: str,
    id_text: Optional[str] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_NEW,
    paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
    preserve_paragraph_styles: Optional[Sequence[str]] = None,
    update_block_handlers: Optional[list[UsfmUpdateBlockHandler]] = None,
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
        assert target_line.strip() == truth_line.strip()
