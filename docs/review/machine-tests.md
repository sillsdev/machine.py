# Machine Tests Review

*Review tests as evidence for changed behavior, edge cases, contracts, and
resource boundaries.*

Governs `tests/**/*.py`.

- A test must exercise the changed behavior, not merely execute the changed
  function. Identify the branches, guards, ordering, error paths, and limits in
  the production diff, and name which test covers each.
- Prefer a focused test. Keep fixtures deterministic: no sleeps, no ambient
  machine state, no dependence on filesystem ordering.
- For string, token, marker, and USFM behavior, include the Unicode, empty,
  malformed, and nested cases. Assert over the parsed structure where the
  contract is structural; a comparison of rendered text can pass while the
  structure underneath is wrong.
- A test that only round-trips input to output proves serialization, not
  behavior. When the change is about what a row or block carries, assert on that
  value directly.
- Name the test that proves the change.
