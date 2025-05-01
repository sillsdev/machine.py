from __future__ import annotations

from copy import copy
from typing import List, Sequence

from ..jobs.translation_file_service import PretranslationInfo
from ..tokenization import LatinWordTokenizer
from ..translation import WordAlignmentMatrix
from .aligned_word_pair import AlignedWordPair
from .scripture_update_block import ScriptureUpdateBlock
from .scripture_update_block_handler import ScriptureUpdateBlockHandler
from .scripture_update_element import ScriptureUpdateElement, ScriptureUpdateElementType
from .usfm_token import UsfmToken, UsfmTokenType

TOKENIZER = LatinWordTokenizer()


class PlaceMarkersScriptureUpdateBlockHandler(ScriptureUpdateBlockHandler):

    def __init__(self, pt_info: Sequence[PretranslationInfo]):
        # TODO: when will len(refs) be >1?
        self._pt_info = {info["refs"][0]: info for info in pt_info}

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        # Nothing to do if there are no markers to place, no alignment to use, or if the block represents an embed
        if (
            len(block.elements) == 0
            or str(block.ref) not in self._pt_info.keys()
            or len(self._pt_info[str(block.ref)]["alignment"]) == 0
            or block.elements[0].type == ScriptureUpdateElementType.EMBED
            or not any(
                (
                    element.type in [ScriptureUpdateElementType.PARAGRAPH, ScriptureUpdateElementType.STYLE]
                    and not element.marked_for_removal
                )
                for element in block.elements[1:]
            )
        ):
            return block

        # Parsing the block's elements potentially involves removing elements so they are not processed twice,
        # but the original block may need to be returned if the two versions of the source/target text to not match up
        orig_elements = copy(block.elements)

        src_sent = ""
        trg_sent = ""
        to_place = []
        src_marker_idxs = []
        placed_elements = [orig_elements[0]]
        end_elements = []
        ignored_elements = []

        # Paragraph markers at the end of the block should stay there
        for i, element in reversed(list(enumerate(orig_elements))):
            if element.type == ScriptureUpdateElementType.PARAGRAPH:
                end_elements.insert(0, element)
                orig_elements.pop(i)
            elif element.type != ScriptureUpdateElementType.EMBED_BLOCK:
                break

        for element in orig_elements[1:]:
            if element.type == ScriptureUpdateElementType.EXISTING_TEXT:
                src_sent += element.tokens[0].to_usfm()
            if element.type == ScriptureUpdateElementType.INSERTED_TEXT:
                trg_sent += element.tokens[0].to_usfm()

            if element.marked_for_removal or element.type == ScriptureUpdateElementType.EMBED_BLOCK:
                ignored_elements.append(element)
            elif element.type in [ScriptureUpdateElementType.PARAGRAPH, ScriptureUpdateElementType.STYLE]:
                to_place.append(element)
                src_marker_idxs.append(len(src_sent))

        src_toks = self._pt_info[str(block.ref)]["source_toks"]
        trg_toks = self._pt_info[str(block.ref)]["translation_toks"]

        # Don't do anything if the source sentence or pretranslation has changed
        if (
            list(t for t in TOKENIZER.tokenize(src_sent)) != src_toks
            or list(t for t in TOKENIZER.tokenize(trg_sent)) != trg_toks  # could just use translation for trg
        ):
            return block

        src_tok_starts = []
        for tok in src_toks:
            src_tok_starts.append(src_sent.index(tok, src_tok_starts[-1] + 1 if len(src_tok_starts) > 0 else 0))
        trg_tok_starts = []
        for tok in trg_toks:
            trg_tok_starts.append(trg_sent.index(tok, trg_tok_starts[-1] + 1 if len(trg_tok_starts) > 0 else 0))

        # Get index of the text token immediately following each marker
        # and predict the corresponding token on the target side
        adj_src_toks = []
        for idx in src_marker_idxs:
            for i, start_idx in reversed(list(enumerate(src_tok_starts))):
                if start_idx < idx:
                    adj_src_toks.append(i + 1)
                    break
                if i == 0:
                    adj_src_toks.append(i)

        alignment = to_word_alignment_matrix(self._pt_info[str(block.ref)]["alignment"])
        adj_trg_toks = [
            self._predict_marker_location(alignment, adj_src_tok, src_toks, trg_toks) for adj_src_tok in adj_src_toks
        ]

        # Collect the markers to be inserted
        to_insert = []
        for element, adj_trg_tok in zip(to_place, adj_trg_toks):
            trg_str_idx = trg_tok_starts[adj_trg_tok] if adj_trg_tok < len(trg_tok_starts) else len(trg_sent)

            # Determine the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_pos = 0
            while insert_pos < len(to_insert) and to_insert[insert_pos][0] <= trg_str_idx:
                insert_pos += 1
            to_insert.insert(insert_pos, (trg_str_idx, element))

        # Construct new text tokens to put between markers
        placed_elements.append(
            ScriptureUpdateElement(
                ScriptureUpdateElementType.INSERTED_TEXT,
                [UsfmToken(UsfmTokenType.TEXT, text=trg_sent[: to_insert[0][0]] if len(to_insert) > 0 else trg_sent)],
            )
        )
        for j, (insert_idx, element) in enumerate(to_insert):
            placed_elements.append(element)
            text_token = UsfmToken(
                UsfmTokenType.TEXT,
                text=(trg_sent[insert_idx : to_insert[j + 1][0]] if j + 1 < len(to_insert) else trg_sent[insert_idx:]),
            )
            placed_elements.append(
                ScriptureUpdateElement(
                    ScriptureUpdateElementType.INSERTED_TEXT,
                    [text_token],
                )
            )

        block._elements = placed_elements + end_elements + ignored_elements
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
                # TODO: experiment w/ using adj_src_tok instead of src_hyp
                # probably doesn't work well w/ word order switches, e.g. eng vs spa noun/adj
                # one issue it does fix is markers getting sucked to punctuation
                # (could be the source of some of the \w\w* issues)
                num_crossings = num_align_crossings(adj_src_tok, trg_hyp)
                if num_crossings < best_num_crossings:
                    best_hyp = trg_hyp
                    best_num_crossings = num_crossings
                if num_crossings == 0:
                    break

        # If no alignments found, insert at the end of the sentence
        return best_hyp if best_hyp != -1 else len(trg_toks)


# TODO: use implementation in jobs.eflomal_aligner when available
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
