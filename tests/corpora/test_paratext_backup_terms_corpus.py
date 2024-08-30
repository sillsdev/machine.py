from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from testutils.corpora_test_helpers import create_test_paratext_backup

from machine.corpora import ParatextBackupTermsCorpus, TextRow


def test_create_corpus():
    temp_dir = TemporaryDirectory()
    backup_dir = create_test_paratext_backup(Path(temp_dir.name))
    corpus = ParatextBackupTermsCorpus(backup_dir, ["PN"], True)
    rows: List[TextRow] = list(corpus.get_rows())
    assert len(rows) == 1
    assert rows[0].text == "Xerxes"
