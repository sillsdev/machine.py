from typing import List

from testutils.corpora_test_helpers import scripture_ref

from machine.corpora import ScriptureRef, TextRow, UsxMemoryText


def test_get_rows_descriptive_title() -> None:
    rows = get_rows(
        r"""<usx version="3.0">
  <book code="MAT" style="id">- Test</book>
  <chapter number="1" style="c" />
  <para style="d">
    <verse number="1" style="v" sid="MAT 1:1" />Descriptive title</para>
  <para style="p">
    The rest of verse one.<verse eid="MAT 1:1" />
    <verse number="2" style="v" />This is verse two.</para>
</usx>
"""
    )
    assert len(rows) == 2

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1"), str.join(",", [str(tr.ref) for tr in rows])
    assert rows[0].text == "Descriptive title", str.join(",", [tr.text for tr in rows])


def test_get_rows_table() -> None:
    rows = get_rows(
        r"""<usx version="3.0">
  <book code="MAT" style="id">- Test</book>
  <chapter number="1" style="c" />
  <table>
    <row style="tr">
      <cell style="tc1" align="start"><verse number="1" style="v" />Chapter</cell>
      <cell style="tcr2" align="end">1</cell>
      <cell style="tc3" align="start">verse</cell>
      <cell style="tcr4" align="end">1</cell>
    </row>
    <row style="tr">
      <cell style="tc1" colspan="2" align="start"><verse number="2" style="v" /></cell>
      <cell style="tc3" colspan="2" align="start">Chapter 1 verse 2</cell>
    </row>
  </table>
</usx>
"""
    )

    assert len(rows) == 2

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1")
    assert rows[0].text == "Chapter 1 verse 1"

    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:2")
    assert rows[1].text == "Chapter 1 verse 2"


def get_rows(usx: str) -> List[TextRow]:
    text = UsxMemoryText("MAT", usx)
    return list(text.get_rows())
