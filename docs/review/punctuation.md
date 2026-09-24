# Punctuation Analysis Review

*Review quotation-mark and punctuation analysis for Unicode safety, malformed
input, and chapter and verse bookkeeping.*

Governs `machine/punctuation_analysis/**/*.py`.

This area's history is almost entirely crash fixes reported from live data, so
review it for the input that should not have reached the code rather than for
the happy path.

- Index by text element, not by code point. A combining mark or a
  multi-code-point quotation mark must not split.
- Assume the chapter or verse is missing, out of range, or unparsable. The
  extractor runs over real Paratext projects: `53992c8` fixed chapter numbers
  that came back wrong and `ca37757` fixed a crash on an invalid chapter.
- A depth or state machine that tracks open and close marks must terminate on
  unbalanced input and say what it saw, rather than running to the end of the
  text.
- Treat quotation-mark identity as exact. Denormalization maps one code point to
  another; a case-insensitive or locale-aware comparison here is a bug.
- Quote convention detection reads whole projects and must degrade rather than
  raise when a book is absent or unreadable (`0c3cd9c`, `d93dacd`).
- Add a fixture for the malformed case with the fix. Every fix above came from
  live data, not from review.
