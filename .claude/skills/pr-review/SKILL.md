---
name: pr-review
description: Review a pull request in sillsdev/machine.py - find and verify with code-review, then post short numbered line comments with evidence and severity.
argument-hint: "[optional PR number, branch, or review focus]"
user-invocable: true
---

# Writing a machine.py review

Post one short comment per finding, anchored on the line it is about, then one
summary comment. A review is read-only: do not edit, commit, push, or resolve
threads.

Unless verified findings are already in hand, get them first with
`/code-review high <target>`, without `--comment`: it finds and verifies, and
this skill decides what gets posted.

## 1. One finding, one comment

Anchor it on the line. Two problems on one line are two comments. A reviewer
scrolling the diff should meet each point where it applies.

Number findings `F1`, `F2`, ... in the order you post them, and put the number
right after the keyword, so a reply or a later review can refer to one without
quoting it. Not `#1`: GitHub links that to issue 1. Numbers are stable - a
withdrawn finding keeps its number, and a later round continues the sequence.

## 2. Lead with the claim

After the keyword and number, the first sentence names the defect. Evidence
second, fix third, if it fits.

```
Major: F1. Verse range collapses to one row: `_advance_rows` keeps only the
last match, so a `\v 1-2` matched by two rows loses the first row's metadata.
Collect a list.
```

Three lines is long. A finding needing more is a design question - raise it in
the summary instead.

## 3. Label the severity

Every finding is **Critical**, **Important**, or **Low**. Critical means
demonstrated: a failing command, a broken contract, a missing gate. A worry is
not Critical. Important needs an answer from the author; Low is worth knowing
and needs none.

A finding comment posted to the PR starts with the Reviewable keyword for its
severity, followed by a colon. Reviewable reads it and sets the discussion's
disposition:

| Severity | Comment starts with | Disposition in Reviewable |
| --- | --- | --- |
| Critical | `Major:` | Blocking, until a maintainer dismisses it |
| Important | `Minor:` | Discussing, open until the author answers |
| Low | `FYI:` | Informing, starts resolved |

Use the keywords there and nowhere else. The summary, replies, and a review
that is not posted use the severity names: `Minor` reads as trivial, and an
Important finding is not. A finding comment that starts with any other word gets
Reviewable's default, which for a reviewer is Blocking.

## 4. Carry the evidence

Every comment gets a `path:line` and a consequence. Mark what you did not
confirm `Unverified`; an unverified concern is never Critical.

- Do not report pre-existing issues the diff does not touch.
- Do not ask for a migration, modernization, or benchmark the diff gave no
  reason for.
- A search that found nothing proves absence only if you state what you
  searched.
- Name the commands you ran and what they returned.
- Reproduce before you report. The environment can run the code: a finding you
  tried and failed to reproduce is worth more than one you only reasoned about.
- A coverage percentage is not evidence that a changed line is tested.

## 5. Close with five lines

1. Verdict: approve, approve with fixes, or request changes.
2. The one thing that matters most, with its number and `path:line`.
3. Counts by severity.
4. What you ran, and its result.
5. What you could not verify.

Say which public API, optional dependency, published-wheel surface, or parity
contract with `sillsdev/machine` changed, or `None verified`.

Then mark each finding, by number, **changed**, **accepted**, or
**unverified**. Leave nothing implicit: a thread with no follow-up leaves nobody
able to tell which findings mattered.

For an adversarial second pass, apply `docs/review/devils-advocate.md`.
