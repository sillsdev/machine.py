# Machine Devil's Advocate

*Optional adversarial second pass for a high-risk change.*

Read the merge-base diff, the normal review output, and the tests. Change
nothing.

This pass prunes the first review; it does not sweep for new categories. Take
the findings it produced and make each earn its place.

The threshold: a concern is a finding only when the evidence is already in hand.
Label each item `Verified objection` or `Unverified question`, and say what
evidence would settle an unverified one instead of supplying the answer. Drop
what survives neither label.

Challenge the most consequential claim first, one objection at a time, each with
a `path:line` and a concrete scenario. In this repository the claims that have
failed before are:

- a USFM or reference change whose test proves the happy path only;
- a parity claim about `sillsdev/machine` with no checked comparison;
- a public API change called internal because every caller in this repo still
  works;
- a build-job change asserted to be safe under failure without a run that failed.

Close with `Top risk`, `Evidence still needed`, and `Would this block merge?`.
