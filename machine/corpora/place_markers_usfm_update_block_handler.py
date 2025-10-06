from __future__ import annotations

from typing import List, TypedDict, cast

from ..translation.word_alignment_matrix import WordAlignmentMatrix
from .update_usfm_parser_handler import UpdateUsfmMarkerBehavior
from .usfm_token import UsfmToken, UsfmTokenType
from .usfm_update_block import UsfmUpdateBlock
from .usfm_update_block_element import UsfmUpdateBlockElement, UsfmUpdateBlockElementType
from .usfm_update_block_handler import UsfmUpdateBlockHandler, UsfmUpdateBlockHandlerError

PLACE_MARKERS_ALIGNMENT_INFO_KEY = "alignment_info"


class PlaceMarkersAlignmentInfo(TypedDict):
    source_tokens: List[str]
    translation_tokens: List[str]
    alignment: WordAlignmentMatrix
    paragraph_behavior: UpdateUsfmMarkerBehavior
    style_behavior: UpdateUsfmMarkerBehavior


class PlaceMarkersUsfmUpdateBlockHandler(UsfmUpdateBlockHandler):

    def process_block(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock:
        elements = list(block.elements)

        # Nothing to do if there are no markers to place or no alignment to use
        if PLACE_MARKERS_ALIGNMENT_INFO_KEY not in block.metadata:
            return block

        alignment_info = cast(PlaceMarkersAlignmentInfo, block.metadata[PLACE_MARKERS_ALIGNMENT_INFO_KEY])
        if (
            len(elements) == 0
            or alignment_info["alignment"].row_count == 0
            or alignment_info["alignment"].column_count == 0
            or not any(
                (
                    (
                        e.type == UsfmUpdateBlockElementType.PARAGRAPH
                        and alignment_info["paragraph_behavior"] == UpdateUsfmMarkerBehavior.PRESERVE
                        and len(e.tokens) == 1
                    )
                    or (
                        e.type == UsfmUpdateBlockElementType.STYLE
                        and alignment_info["style_behavior"] == UpdateUsfmMarkerBehavior.PRESERVE
                    )
                )
                for e in elements
            )
        ):
            return block

        # Paragraph markers at the end of the block should stay there
        # Section headers should be ignored but re-inserted in the same position relative to other paragraph markers
        end_elements = []
        eob_empty_paras = True
        header_elements = []
        para_markers_left = 0
        for i, element in reversed(list(enumerate(elements))):
            if element.type == UsfmUpdateBlockElementType.PARAGRAPH and not element.marked_for_removal:
                if len(element.tokens) > 1:
                    header_elements.insert(0, (para_markers_left, element))
                    elements.pop(i)
                else:
                    para_markers_left += 1

                    if eob_empty_paras:
                        end_elements.insert(0, element)
                        elements.pop(i)
            elif not (
                element.type == UsfmUpdateBlockElementType.EMBED
                or (element.type == UsfmUpdateBlockElementType.TEXT and len(element.tokens[0].to_usfm().strip()) == 0)
            ):
                eob_empty_paras = False

        src_toks: List[str] = alignment_info["source_tokens"]
        trg_toks: List[str] = alignment_info["translation_tokens"]
        src_tok_idx = 0

        src_sent = ""
        trg_sent = ""
        to_place = []
        adj_src_toks = []
        placed_elements = []
        embed_elements = []
        ignored_elements = []
        for element in elements:
            if element.type == UsfmUpdateBlockElementType.TEXT:
                if element.marked_for_removal:
                    text = element.tokens[0].to_usfm()
                    src_sent += text

                    # Track seen tokens
                    while src_tok_idx < len(src_toks) and src_toks[src_tok_idx] in text:
                        text = text[text.index(src_toks[src_tok_idx]) + len(src_toks[src_tok_idx]) :]
                        src_tok_idx += 1
                    # Handle tokens split across text elements
                    if len(text.strip()) > 0:
                        src_tok_idx += 1
                else:
                    trg_sent += element.tokens[0].to_usfm()

            if element.marked_for_removal or (
                element.type == UsfmUpdateBlockElementType.PARAGRAPH
                and alignment_info["paragraph_behavior"] == UpdateUsfmMarkerBehavior.STRIP
            ):
                ignored_elements.append(element)
            elif element.type == UsfmUpdateBlockElementType.EMBED:
                embed_elements.append(element)
            elif element.type in [UsfmUpdateBlockElementType.PARAGRAPH, UsfmUpdateBlockElementType.STYLE]:
                to_place.append(element)
                adj_src_toks.append(src_tok_idx)

        if len(trg_sent.strip()) == 0:
            return block

        trg_tok_starts = []
        prev_len = 0
        for tok in trg_toks:
            try:
                index_of_trg_tok_in_sent = trg_sent.index(
                    tok, trg_tok_starts[-1] + prev_len if len(trg_tok_starts) > 0 else 0
                )
            except ValueError:
                raise UsfmUpdateBlockHandlerError(
                    block,
                    f'No token "{tok}" found in text "{trg_sent}" at or beyond index'
                    f"{trg_tok_starts[-1] + prev_len if len(trg_tok_starts) > 0 else 0}."
                    "Is the versification correctly specified?",
                )
            trg_tok_starts.append(index_of_trg_tok_in_sent)
            prev_len = len(tok)

        # Predict marker placements and get insertion order
        to_insert = []
        for element, adj_src_tok in zip(to_place, adj_src_toks):
            adj_trg_tok = self._predict_marker_location(alignment_info["alignment"], adj_src_tok, src_toks, trg_toks)

            if (
                adj_trg_tok > 0
                and element.type == UsfmUpdateBlockElementType.STYLE
                and element.tokens[0].marker[-1] == "*"
            ):
                # Insert end tokens directly after the token they follow
                trg_str_idx = trg_tok_starts[adj_trg_tok - 1] + len(trg_toks[adj_trg_tok - 1])
            elif adj_trg_tok < len(trg_tok_starts):
                trg_str_idx = trg_tok_starts[adj_trg_tok]
            else:
                trg_str_idx = len(trg_sent)

            to_insert.append((trg_str_idx, element))
        to_insert.sort(key=lambda x: x[0])
        to_insert += [(len(trg_sent), element) for element in embed_elements + end_elements]

        # Construct new text tokens to put between markers
        # and reincorporate headers and empty end-of-verse paragraph markers
        if to_insert[0][0] > 0:
            placed_elements.append(
                UsfmUpdateBlockElement(
                    UsfmUpdateBlockElementType.TEXT, [UsfmToken(UsfmTokenType.TEXT, text=trg_sent[: to_insert[0][0]])]
                )
            )
        for j, (insert_idx, element) in enumerate(to_insert):
            if element.type == UsfmUpdateBlockElementType.PARAGRAPH:
                while len(header_elements) > 0 and header_elements[0][0] == para_markers_left:
                    placed_elements.append(header_elements.pop(0)[1])
                para_markers_left -= 1

            placed_elements.append(element)
            if insert_idx < len(trg_sent) and (j + 1 == len(to_insert) or insert_idx < to_insert[j + 1][0]):
                if j + 1 < len(to_insert):
                    text_token = UsfmToken(UsfmTokenType.TEXT, text=(trg_sent[insert_idx : to_insert[j + 1][0]]))
                else:
                    text_token = UsfmToken(UsfmTokenType.TEXT, text=(trg_sent[insert_idx:]))
                placed_elements.append(UsfmUpdateBlockElement(UsfmUpdateBlockElementType.TEXT, [text_token]))
        while len(header_elements) > 0:
            placed_elements.append(header_elements.pop(0)[1])

        block._elements = placed_elements + ignored_elements
        return block

    def _predict_marker_location(
        self,
        alignment: WordAlignmentMatrix,
        adj_src_tok: int,
        src_toks: List[str],
        trg_toks: List[str],
    ) -> int:
        # Gets the number of alignment pairs that "cross the line" between
        # the src marker position and the potential trg marker position, (src_idx - .5) and (trg_idx - .5)
        def num_align_crossings(src_idx: int, trg_idx: int) -> int:
            crossings = 0
            for i in range(alignment.row_count):
                for j in range(alignment.column_count):
                    if alignment[i, j] and ((i < src_idx and j >= trg_idx) or (i >= src_idx and j < trg_idx)):
                        crossings += 1
            return crossings

        # If the token on either side of a potential target location is punctuation,
        # use it as the basis for deciding the target marker location
        trg_hyp = -1
        punct_hyps = [-1, 0]
        for punct_hyp in punct_hyps:
            src_hyp = adj_src_tok + punct_hyp
            if src_hyp < 0 or src_hyp >= len(src_toks):
                continue
            # Only accept aligned pairs where both the src and trg token are punctuation
            hyp_tok = src_toks[src_hyp]
            if len(hyp_tok) > 0 and not any(c.isalpha() for c in hyp_tok) and src_hyp < alignment.row_count:
                aligned_trg_toks = list(alignment.get_row_aligned_indices(src_hyp))
                # If aligning to a token that precedes that marker,
                # the trg token predicted to be closest to the marker
                # is the last token aligned to the src rather than the first
                for trg_idx in reversed(aligned_trg_toks) if punct_hyp < 0 else aligned_trg_toks:
                    trg_tok = trg_toks[trg_idx]
                    if len(trg_tok) > 0 and not any(c.isalpha() for c in trg_tok):
                        trg_hyp = trg_idx
                        break
            if trg_hyp != -1:
                # Since the marker location is represented by the token after the marker,
                # adjust the index when aligning to punctuation that precedes the token
                return trg_hyp + (1 if punct_hyp == -1 else 0)

        hyps = [0, 1, 2]
        best_hyp = -1
        best_num_crossings = 200**2  # mostly meaningless, a big number
        checked = set()
        for hyp in hyps:
            src_hyp = adj_src_tok + hyp
            if src_hyp in checked:
                continue
            trg_hyp = -1
            while trg_hyp == -1 and src_hyp >= 0 and src_hyp < alignment.row_count:
                checked.add(src_hyp)
                aligned_trg_toks = list(alignment.get_row_aligned_indices(src_hyp))
                if len(aligned_trg_toks) > 0:
                    # If aligning with a source token that precedes the marker,
                    # the target token predicted to be closest to the marker is the last aligned token rather than the first
                    trg_hyp = aligned_trg_toks[-1 if hyp < 0 else 0]
                else:  # continue the search outwards
                    src_hyp += -1 if hyp < 0 else 1
            if trg_hyp != -1:
                num_crossings = num_align_crossings(adj_src_tok, trg_hyp)
                if num_crossings < best_num_crossings:
                    best_hyp = trg_hyp
                    best_num_crossings = num_crossings
                if num_crossings == 0:
                    break

        # If no alignments found, insert at the end of the sentence
        return best_hyp if best_hyp != -1 else len(trg_toks)
