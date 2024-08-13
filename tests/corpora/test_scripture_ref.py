import unittest

from machine.scripture import ScriptureRef


class TestScriptureRef(unittest.TestCase):

    def compare_to_strict(self, ref1_str, ref2_str):
        ref1 = ScriptureRef.parse(ref1_str)
        ref2 = ScriptureRef.parse(ref2_str)

        result = ref1.compare_to(ref2)

        if result < 0:
            result = -1
        elif result > 0:
            result = 1
        return result

    def compare_to_relaxed(self, ref1_str, ref2_str):
        ref1 = ScriptureRef.parse(ref1_str)
        ref2 = ScriptureRef.parse(ref2_str)

        result = ref1.compare_to(ref2, strict=False)

        if result < 0:
            result = -1
        elif result > 0:
            result = 1
        return result

    def test_compare_to_strict(self):
        self.assertEqual(self.compare_to_strict("MAT 1:1", "MAT 1:2"), -1, "VerseLessThan")
        self.assertEqual(self.compare_to_strict("MAT 1:1", "MAT 1:1"), 0, "VerseEqualTo")
        self.assertEqual(self.compare_to_strict("MAT 1:2", "MAT 1:1"), 1, "VerseGreaterThan")
        self.assertEqual(self.compare_to_strict("MAT 1:0/1:p", "MAT 1:0/2:p"), -1, "NonVerseLessThan")
        self.assertEqual(self.compare_to_strict("MAT 1:0/1:p", "MAT 1:0/1:p"), 0, "NonVerseEqualTo")
        self.assertEqual(self.compare_to_strict("MAT 1:0/2:p", "MAT 1:0/1:p"), 1, "NonVerseGreaterThan")
        self.assertEqual(self.compare_to_strict("MAT 1:0/1:esb", "MAT 1:0/1:esb/1:p"), -1, "NonVerseParentChild")

    def test_compare_to_relaxed(self):
        self.assertEqual(self.compare_to_relaxed("MAT 1:1", "MAT 1:2"), -1, "VerseLessThan")
        self.assertEqual(self.compare_to_relaxed("MAT 1:1", "MAT 1:1"), 0, "VerseEqualTo")
        self.assertEqual(self.compare_to_relaxed("MAT 1:2", "MAT 1:1"), 1, "VerseGreaterThan")
        self.assertEqual(
            self.compare_to_relaxed("MAT 1:0/1:p", "MAT 1:0/2:p"), 0, "NonVerseSameMarkerDifferentPosition"
        )
        self.assertEqual(self.compare_to_relaxed("MAT 1:0/2:esb", "MAT 1:0/1:esb/1:p"), -1, "NonVerseParentChild")
