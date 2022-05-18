from typing import Iterable, List, Sequence, Tuple


def batch(
    segments: Iterable[Tuple[Sequence[str], Sequence[str]]], batch_size: int
) -> Iterable[Tuple[List[Sequence[str]], List[Sequence[str]]]]:
    src_segments: List[Sequence[str]] = []
    trg_segments: List[Sequence[str]] = []
    for source_segment, target_segment in segments:
        src_segments.append(source_segment)
        trg_segments.append(target_segment)
        if len(src_segments) == batch_size:
            yield src_segments, trg_segments
            src_segments.clear()
            trg_segments.clear()
    if len(src_segments) > 0:
        yield src_segments, trg_segments
