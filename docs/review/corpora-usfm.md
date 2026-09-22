# Corpora and USFM Review

*Review corpus and USFM changes for deterministic marker and token behavior,
ScriptureRef and versification correctness, Unicode handling, and safe file
inputs.*

Governs `machine/corpora/**/*.py`.

- Trace changed behavior from source text or corpus row through tokenization,
  parsing, ScriptureRef and versification conversion, update handling, and
  emitted text. A round trip that produces the same string is not proof that the
  intermediate structure is right.
- Marker and paragraph handling is the defect class that recurs most here.
  Unclosed and implicitly closed character styles (`29088f8`, `962ca36`), an
  element in an unexpected position (`9868016`), spacing around end markers
  (`417c95f`), and paragraph markers in row text (`b07fcb6`) were all shipped
  bugs. For a change that touches any of them, assert over marker nesting, empty
  input, and malformed input, not only the happy path.
- Reference and versification arithmetic is the next most common
  (`29d4dbb`). Check that a verse range, a segment path, and a versification
  change resolve or fail according to the existing contract, and assert over the
  affected book, chapter, and verse mapping rather than over rendered text.
- Compare markers, tokens, and identifiers by exact string identity. Case
  folding and locale-aware normalization belong only where the operation is
  genuinely linguistic; quotation-mark identity is not.
- For files, ZIPs, and streams, preserve entry and byte limits, path validation,
  closing, and actionable errors. Be explicit about who owns a handle passed in.
- A row that produces no text still carries metadata. A verse range matched by
  several rows must not silently collapse to one of them.
