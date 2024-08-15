from machine.corpora import ScriptureRef


def test_compare_to_strict():
    assert compare_to_strict("MAT 1:1", "MAT 1:2") == -1, "VerseLessThan"
    assert compare_to_strict("MAT 1:1", "MAT 1:1") == 0, "VerseEqualTo"
    assert compare_to_strict("MAT 1:2", "MAT 1:1") == 1, "VerseGreaterThan"
    assert compare_to_strict("MAT 1:0/1:p", "MAT 1:0/2:p") == -1, "NonVerseLessThan"
    assert compare_to_strict("MAT 1:0/1:p", "MAT 1:0/1:p") == 0, "NonVerseEqualTo"
    assert compare_to_strict("MAT 1:0/2:p", "MAT 1:0/1:p") == 1, "NonVerseGreaterThan"
    assert compare_to_strict("MAT 1:0/1:esb", "MAT 1:0/1:esb/1:p") == -1, "NonVerseParentChild"


def test_compare_to_relaxed():
    assert compare_to_relaxed("MAT 1:1", "MAT 1:2") == -1, "VerseLessThan"
    assert compare_to_relaxed("MAT 1:1", "MAT 1:1") == 0, "VerseEqualTo"
    assert compare_to_relaxed("MAT 1:2", "MAT 1:1") == 1, "VerseGreaterThan"
    assert compare_to_relaxed("MAT 1:0/1:p", "MAT 1:0/2:p") == 0, "NonVerseSameMarkerDifferentPosition"
    assert compare_to_relaxed("MAT 1:0/2:esb", "MAT 1:0/1:esb/1:p") == -1, "NonVerseParentChild"


def compare_to_strict(ref1_str, ref2_str):
    ref1 = ScriptureRef.parse(ref1_str)
    ref2 = ScriptureRef.parse(ref2_str)

    result = ref1.compare_to(ref2)

    if result < 0:
        result = -1
    elif result > 0:
        result = 1
    return result


def compare_to_relaxed(ref1_str, ref2_str):
    ref1 = ScriptureRef.parse(ref1_str)
    ref2 = ScriptureRef.parse(ref2_str)

    result = ref1.compare_to(ref2, strict=False)

    if result < 0:
        result = -1
    elif result > 0:
        result = 1
    return result
