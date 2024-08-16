from pytest import raises

from machine.corpora import ScriptureRef


def test_compare_to_strict():
    assert compare_to_strict("MAT 1:1", "MAT 1:2") == -1, "VerseLessThan"
    assert compare_to_strict("MAT 1:1", "MAT 1:1") == 0, "VerseEqualTo"
    assert compare_to_strict("MAT 1:2", "MAT 1:1") == 1, "VerseGreaterThan"
    assert compare_to_strict("MAT 1:1-3", "MAT 1:1") == 1, "MultiVerseExtensionGreaterThan"
    assert compare_to_strict("MAT 1:1", "MAT 1:1-3") == -1, "MultiVerseExtensionLessThan"
    assert compare_to_strict("MAT 1:1-3", "MAT 1:2") == -1, "MultiVerseStartLessThan"
    assert compare_to_strict("MAT 1:2", "MAT 1:1-3") == 1, "MultiVerseEndGreaterThan"
    assert compare_to_strict("MAT 1:0/1:p", "MAT 1:0/2:p") == -1, "NonVerseLessThan"
    assert compare_to_strict("MAT 1:0/1:p", "MAT 1:0/1:p") == 0, "NonVerseEqualTo"
    assert compare_to_strict("MAT 1:0/2:p", "MAT 1:0/1:p") == 1, "NonVerseGreaterThan"
    assert compare_to_strict("MAT 1:0/1:esb", "MAT 1:0/1:esb/1:p") == -1, "NonVerseParentChild"
    assert compare_to_strict("MAT 1:0/2:esb", "MAT 1:0/1:esb/1:p") == 1, "NonVerseParentOtherChild"


def test_compare_to_relaxed():
    assert compare_to_relaxed("MAT 1:1", "MAT 1:2") == -1, "VerseLessThan"
    assert compare_to_relaxed("MAT 1:1", "MAT 1:1") == 0, "VerseEqualTo"
    assert compare_to_relaxed("MAT 1:2", "MAT 1:1") == 1, "VerseGreaterThan"
    assert compare_to_relaxed("MAT 1:0/1:p", "MAT 1:0/2:p") == 0, "NonVerseSameMarkerDifferentPosition"
    assert compare_to_relaxed("MAT 1:0/1:esb", "MAT 1:0/1:esb/1:p") == -1, "NonVerseParentChild"
    assert compare_to_relaxed("MAT 1:0/2:esb", "MAT 1:0/1:esb/1:p") == -1, "NonVerseParentOtherChild"


def test_is_equal_to():
    ref1 = ScriptureRef.parse("MAT 1:1/1:p")
    ref1dup = ScriptureRef.parse("MAT 1:1/1:p")
    ref2 = ScriptureRef.parse("MAT 1:2/1:p")
    obj1 = "A different type"

    assert ref1 == ref1dup
    assert ref1 != ref2
    assert ref1 != obj1


def test_is_equal_to_throws_argument_exception():
    ref1 = ScriptureRef.parse("MAT 1:1/1:p")
    obj1 = "A different type"

    with raises(TypeError):
        ref1.compare_to(obj1)


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
