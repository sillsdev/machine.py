---
name: commit-messages
description: How to write a commit message in sillsdev/machine.py - imperative subject, short body.
---

# Commit messages

Name the change in an imperative, sentence-case subject under about 72
characters, with no terminal punctuation:

- `Fix unclosed style marker crash (#364)`
- `Port the marker placement unit test from sillsdev/machine#496 (#367)`

A body, when there is one: blank line after the subject, wrapped at about 80
columns, saying what changed and why. Reference a GitHub issue when one exists.

## Two traps in the history

1. The `(#N)` suffix is added by GitHub when a pull request is squashed. Never
   type it into a local commit.
2. A handful of old commits carry a Jira identifier such as `LT-22605`. That is
   historical; use a GitHub issue reference.

The 72-character limit is not enforced and subjects over 100 characters exist.
Do not rewrite shared history to satisfy it, or to fix a message on a pushed
branch - add a corrective commit unless the author asks for the rewrite.
