from io import StringIO
from typing import Dict, List, Optional

from testutils.memory_paratext_project_file_handler import DefaultParatextProjectSettings
from testutils.memory_paratext_project_versification_error_detector import (
    MemoryParatextProjectVersificationErrorDetector,
)

from machine.corpora import ParatextProjectSettings, UsfmVersificationError, UsfmVersificationErrorType
from machine.scripture import ORIGINAL_VERSIFICATION, Versification


def get_usfm_versification_errors_no_errors():
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
    assert len(env.get_usfm_versification_errors()) == 0


def get_usfm_versification_errors_missing_verses():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.MISSING_VERSE


def get_usfm_versification_missing_chapter():
    env = _TestEnvironment(
        files={
            "653JNTest.SFM": r"""\id 3JN
    """
        }
    )
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.MISSING_CHAPTER


def get_usfm_versification_errors_extra_verse():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.EXTRA_VERSE


def get_usfm_versification_errors_invalid_verse():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.INVALID_VERSE_RANGE


def get_usfm_versification_errors_extra_verse_segment():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.EXTRA_VERSE_SEGMENT


def get_usfm_versification_errors_missing_verse_segments():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.MISSING_VERSE_SEGMENT


def get_usfm_versification_errors_ignore_noncanonicals():
    env = _TestEnvironment(
        files={
            "98XXETest.SFM": r"""\id XXE
    \c 1
    \v 3-2
    """
        }
    )
    assert len(env.get_usfm_versification_errors()) == 0


def get_usfm_versification_errors_excluded_in_custom_vrs():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.EXTRA_VERSE


def get_usfm_versification_errors_multiple_books():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 1
    assert errors[0].type == UsfmVersificationErrorType.MISSING_VERSE


def get_usfm_versification_errors_multiple_chapters():
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
    errors = env.get_usfm_versification_errors()
    assert len(errors) == 2
    assert errors[0].type == UsfmVersificationErrorType.MISSING_VERSE
    assert errors[0].type == UsfmVersificationErrorType.EXTRA_VERSE


class _TestEnvironment:
    def __init__(self, settings: Optional[ParatextProjectSettings] = None, files: Optional[Dict[str, str]] = None):
        self._settings = settings
        self._files = files or {}
        self.detector = MemoryParatextProjectVersificationErrorDetector(settings, self._files)

    def get_usfm_versification_errors(self) -> List[UsfmVersificationError]:
        return self.detector.get_usfm_versification_errors()


def get_custom_versification(
    custom_vrs_contents: str, base_versification: Optional[Versification] = None
) -> Versification:
    stream = StringIO(custom_vrs_contents)
    versification = base_versification or Versification("custom", "vers.txt", ORIGINAL_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")
    return versification
