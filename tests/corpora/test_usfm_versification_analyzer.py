from io import StringIO
from typing import Dict, List, Optional, Set

from testutils.memory_paratext_project_file_handler import DefaultParatextProjectSettings
from testutils.memory_usfm_versification_analyzer import MemoryUsfmVersificationAnalyzer

from machine.corpora import ParatextProjectSettings, UsfmVersificationAnalysis, UsfmVersificationDiagnosticType
from machine.scripture import ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION, Versification


def test_no_errors():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    \v 15
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert analysis.total_num_encountered_verses == 15
    assert len(analysis.diagnostics) == 0


def test_missing_verse():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 14
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].filename == "653JNTest.SFM"
    assert analysis.diagnostics[0].line_numbers == [16]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:15"

    analysis = env.analyze_usfm_versification(only_chapters={"3JN": set()})
    assert len(analysis.diagnostics) == 0
    assert analysis.total_num_encountered_verses == 0


def test_missing_chapter():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 0
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 15
    assert analysis.diagnostics[0].line_numbers == [1]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:1-15"


def test_extra_verse():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    \v 15
    \v 16
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 16
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.EXTRA
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [18]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:16"


def test_invalid_verse():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 13-12
    \v 14
    \v 15
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 15
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.INVALID
    assert analysis.diagnostics[0].num_affected_verses == 2
    assert analysis.diagnostics[0].line_numbers == [14]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:13-12"


def test_extra_verse_segment():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14a
    \v 14b
    \v 15
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 2
    assert analysis.total_num_encountered_verses == 15
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.INCORRECT_VERSE_SEGMENT
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [16]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:14a"


def test_missing_verse_segment():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    \v 15
    """
        },
        settings=DefaultParatextProjectSettings(versification=get_custom_versification(r"*3JN 1:13,a,b")),
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 15
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.INCORRECT_VERSE_SEGMENT
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [15]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:13"


def test_ignore_noncanonicals():
    env = _TestEnvironment(
        files={
            "98XXETest.SFM": r"""\id XXE
    \c 1
    \v 3-2
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["XXE"])
    assert len(analysis.diagnostics) == 0


def test_extra_verse_excluded_in_custom_vrs():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    \v 15
    """,
        },
        settings=DefaultParatextProjectSettings(versification=get_custom_versification(r"-3JN 1:13")),
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 15
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.EXTRA
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [15]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:13"


def test_multiple_books():
    env = _TestEnvironment(
        files={
            "642JNTest.SFM": r"""\id 2JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    """,
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \v 13
    \v 14
    \v 15
    """,
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["2JN", "3JN"])
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 27
    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [14]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "2JN 1:13"


def test_multiple_chapters():
    env = _TestEnvironment(
        files={
            "642JNTest.SFM": r"""\id 2JN
    \c 1
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6
    \v 7
    \v 8
    \v 9
    \v 10
    \v 11
    \v 12
    \c 2
    \v 1
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["2JN"])
    assert len(analysis.diagnostics) == 2
    assert analysis.total_num_encountered_verses == 13

    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [14]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "2JN 1:13"

    assert analysis.diagnostics[1].type == UsfmVersificationDiagnosticType.EXTRA
    assert analysis.diagnostics[1].num_affected_verses == 1
    assert analysis.diagnostics[1].line_numbers == [16]
    assert len(analysis.diagnostics[1].references) == 1
    assert str(analysis.diagnostics[1].references[0]) == "2JN 2:1"

    analysis = env.analyze_usfm_versification(only_chapters={"2JN": {1}})
    assert len(analysis.diagnostics) == 1
    assert analysis.total_num_encountered_verses == 12

    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [14]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "2JN 1:13"


def test_invalid_chapter_number():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1.
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 2
    assert analysis.total_num_encountered_verses == 0

    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.INVALID
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [2]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN :0"

    assert analysis.diagnostics[1].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[1].num_affected_verses == 15
    assert analysis.diagnostics[1].line_numbers == [1]
    assert len(analysis.diagnostics[1].references) == 1
    assert str(analysis.diagnostics[1].references[0]) == "3JN 1:1-15"


def test_invalid_verse_number():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    \c 1
    \v v1
    """
        }
    )
    analysis = env.analyze_usfm_versification(only_books=["3JN"])
    assert len(analysis.diagnostics) == 2
    assert analysis.total_num_encountered_verses == 1

    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.INVALID
    assert analysis.diagnostics[0].num_affected_verses == 1
    assert analysis.diagnostics[0].line_numbers == [3]
    assert len(analysis.diagnostics[0].references) == 1
    assert str(analysis.diagnostics[0].references[0]) == "3JN 1:v1"

    assert analysis.diagnostics[1].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[1].num_affected_verses == 15
    assert analysis.diagnostics[1].line_numbers == [3]
    assert len(analysis.diagnostics[1].references) == 1
    assert str(analysis.diagnostics[1].references[0]) == "3JN 1:1-15"


def test_unsupported_cross_chapter_verse_reference():
    env = _TestEnvironment(
        files={
            "03LEVTest.SFM": r"""\id LEV
    \c 6
    \v 1
    \v 2
    \v 3
    \v 4
    \v 5
    \v 6-9
    \v 10-30
    """
        },
        # The project uses the English versification, in which LEV 6:6-9 maps to
        # 5:25-6:2 in the Original versification (crossing a chapter boundary).
        settings=DefaultParatextProjectSettings(versification=ENGLISH_VERSIFICATION),
    )
    analysis = env.analyze_usfm_versification(only_books=["LEV"])
    assert len(analysis.diagnostics) == 3
    assert analysis.total_num_encountered_verses == 30
    assert analysis.total_num_affected_verses == 833

    assert analysis.diagnostics[0].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[0].num_affected_verses == 104
    assert analysis.diagnostics[0].line_numbers == [1]
    assert len(analysis.diagnostics[0].references) == 5
    assert str(analysis.diagnostics[0].references[0]) == "LEV 1:1-17"

    assert analysis.diagnostics[1].type == UsfmVersificationDiagnosticType.UNSUPPORTED_VERSE_RANGE
    assert analysis.diagnostics[1].num_affected_verses == 4
    assert analysis.diagnostics[1].line_numbers == [8]
    assert len(analysis.diagnostics[1].references) == 1
    assert str(analysis.diagnostics[1].references[0]) == "LEV 6:6-9"

    assert analysis.diagnostics[2].type == UsfmVersificationDiagnosticType.MISSING
    assert analysis.diagnostics[2].num_affected_verses == 725
    assert analysis.diagnostics[2].line_numbers == [9]
    assert len(analysis.diagnostics[2].references) == 21
    assert str(analysis.diagnostics[2].references[0]) == "LEV 7:1-38"


class _TestEnvironment:
    def __init__(self, settings: Optional[ParatextProjectSettings] = None, files: Optional[Dict[str, str]] = None):
        self._settings = settings
        self._files = files or {}
        self.analyzer = MemoryUsfmVersificationAnalyzer(settings, self._files)

    def analyze_usfm_versification(
        self,
        only_books: Optional[List[str]] = None,
        only_chapters: Optional[Dict[str, Optional[Set[int]]]] = None,
    ) -> UsfmVersificationAnalysis:
        if only_chapters is not None:
            return self.analyzer.analyze_usfm_versification(only_chapters)
        book_ids_and_chapters: Optional[Dict[str, Optional[Set[int]]]] = (
            {book: None for book in only_books} if only_books is not None else None
        )
        return self.analyzer.analyze_usfm_versification(book_ids_and_chapters)


def get_custom_versification(
    custom_vrs_contents: str, base_versification: Optional[Versification] = None
) -> Versification:
    stream = StringIO(custom_vrs_contents)
    versification = base_versification or Versification("custom", "vers.txt", ORIGINAL_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")
    return versification
