from machine.corpora import AlignedWordPair


def test_parse():
    wps = list(AlignedWordPair.from_string("1-0:0.111111:0.222222 0-1:0.222222:0.111111 2-NULL:0.111111:0.222222"))
    assert len(wps) == 3
    assert wps[0].translation_score == 0.111111
    assert wps[0].alignment_score == 0.222222
    wps = list(AlignedWordPair.from_string("1-0:0.111111"))
    assert len(wps) == 1
    assert wps[0].translation_score == 0.111111
    assert wps[0].alignment_score == -1
    wps = list(AlignedWordPair.from_string("1-0"))
    assert len(wps) == 1
    assert wps[0].translation_score == -1
    assert wps[0].alignment_score == -1


def test_parse_to_string():
    wps = [
        AlignedWordPair(0, 1, translation_score=0.1, alignment_score=0.1),
        AlignedWordPair(0, 1, translation_score=0.1, alignment_score=0.1),
    ]
    alignment_string = AlignedWordPair.to_string(wps)
    parsed_wps = list(AlignedWordPair.from_string(alignment_string))
    assert parsed_wps == wps
    assert parsed_wps[0].translation_score == wps[0].translation_score
    assert parsed_wps[0].alignment_score == wps[0].alignment_score
