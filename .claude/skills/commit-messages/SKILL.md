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

The `(#N)` suffix in the history is added by GitHub when a pull request is
squashed. Never type it into a local commit.

Do not rewrite shared history to fix a message on a pushed branch - add a
corrective commit unless the author asks for the rewrite.
