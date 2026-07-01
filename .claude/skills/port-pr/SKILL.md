---
name: port-pr
description: Port a machine (C#) PR into machine.py (Python). Given a GitHub issue number for a porting task in sillsdev/machine.py, finds the linked machine (C#) PR, ports its changes to the Python codebase, runs the local checks (black/flake8/isort/pyright/pytest), and opens a PR that closes the issue. Use when asked to port a PR/issue from machine (C#), complete a "porting" issue, or sync a machine change into machine.py.
---

# Port a machine (C#) PR into machine.py (Python)

`machine` (C#) and `machine.py` (Python) are direct, intentionally-synced ports of each
other. This skill ports a change that already landed in `machine` into `machine.py`,
driven by a "porting" issue in `sillsdev/machine.py`.

**Required argument:** the GitHub issue number in `sillsdev/machine.py` (always exists).

## Repos

- Python target repo: the current working directory (`sillsdev/machine.py`).
- C# source repo: the sibling clone at `../machine` (`sillsdev/machine`).
  Use the local clone for reading surrounding context; use `gh ... --repo sillsdev/machine`
  for authoritative PR data.

## Step 1 — Read the porting issue

```bash
gh issue view <ISSUE> --json title,body,labels
```

- The body looks like: `Port any relevant changes in https://github.com/sillsdev/machine/pull/<PR> from machine to machine.py.`
  Extract `<PR>` — the machine (C#) PR number — from that URL.
- The issue title looks like `Port '<Title>'`. Keep `<Title>` for the branch and PR.
- If the body has no machine (C#) PR link, stop and ask the user for the source PR.

## Step 2 — Understand the source change

```bash
gh pr view <PR> --repo sillsdev/machine --json title,body,files,commits
gh pr diff <PR> --repo sillsdev/machine
```

Read the full diff. For each changed C# file, open the corresponding file(s) in
`../machine` to understand the surrounding context, and identify the Python counterpart
(see mapping below). Read the existing Python code you're about to change so the port
matches local idiom.

Note: not every change ports. Skip C#-only concerns (`.csproj`/`.sln`/`Directory.*.props`,
`AssemblyInfo`, `omnisharp.json`, csharpier/editorconfig formatting, NuGet packaging).
The issue says "any *relevant* changes" — use judgment and call out anything you
intentionally skip.

## Step 3 — File & API mapping

| machine (C#) | machine.py (Python) |
|---|---|
| `src/SIL.Machine/<PascalArea>/<PascalCase>.cs` (or the matching `SIL.Machine.*` project) | `machine/<area>/<snake_case>.py` |
| `tests/SIL.Machine.Tests/<PascalArea>/<PascalCase>Tests.cs` (or the matching `*.Tests` project) | `tests/<area>/test_<snake_case>.py` |
| `PascalCase` methods / `camelCase` locals / `_camelCase` fields | `snake_case` functions/vars |
| `IReadOnlyList<T>` / `IDictionary<,>` / `ISet<T>` etc. | `Sequence[T]`/`list` / `Mapping`/`dict` / `Set`/`set` etc. |
| NUnit `Assert.That(...)` | pytest plain `assert` (check neighboring test files) |
| `AssemblyInfo`/`.csproj` `<Version>` | `pyproject.toml` `version` (poetry) |

The top-level Python areas are: `annotations`, `clusterers`, `corpora`, `jobs`,
`optimization`, `punctuation_analysis`, `scripture`, `sequence_alignment`, `statistics`,
`tokenization`, `translation`, `utils`. Some C# code lives in tool/plugin projects
(`SIL.Machine.Tool`, `SIL.Machine.Translation.Thot`, `SIL.Machine.Morphology.HermitCrab`,
etc.); the Python equivalent may live under `machine/jobs` or a different area, or may not
exist at all. Find the Python counterpart by searching for the type/method name (translated
to snake_case): `grep -ri "<type_or_method_name>" machine tests` before assuming a path.

Port the **behavior**, not the syntax. Match existing Python patterns in the neighboring
code (naming, type hints, dataclasses, generators vs. loops, async conventions). Port the
tests too.

## Step 4 — Branch & apply

Create a branch off `main` (do not commit to `main`):

```bash
git switch main && git pull && git switch -c port-<slug>
```

where `<slug>` is a short kebab-case form of the issue title (e.g.
`port-update-library-version-to-1.8.11`).

Apply the ported changes with Edit/Write.

## Step 5 — Verify locally

Install, format, lint, type-check, and test (this is `local_check.sh`):

```bash
poetry install
poetry run black .
poetry run flake8 .
poetry run isort .
poetry run pyright
poetry run pytest
```

`black` and `isort` rewrite files in place; `flake8` and `pyright` are gates that must pass
clean. Fix any formatting, lint, type, or test failures before proceeding. Report the test
results plainly (pass/fail counts); don't claim success if anything failed.

## Step 6 — Commit, push, open PR (pause first)

Show the user a summary of the diff and the proposed PR title/body, and **confirm before
pushing**. Then:

```bash
git add -A
git commit   # message: Port '<Title>' from machine PR #<PR>
git push -u origin port-<slug>
gh pr create --title "Port '<Title>' from machine" --body "<body>"
```

Commit message footer:

```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

### PR body template

```markdown
Ports [machine PR #<PR>](https://github.com/sillsdev/machine/pull/<PR>) — <one-line summary of the change>.

## <Section per area changed>
<What changed and why, mirroring the source PR. Include short before/after or code snippets where helpful.>

## Tests
<Tests ported / added.>

Closes #<ISSUE>
```

PR body footer:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Notes

- Keep the two codebases as similar as is reasonable for a Python-vs-C# port.
- If the source PR spans multiple commits, the squashed PR diff is the source of truth, but
  reading individual commits can clarify intent.
- If a change has no sensible Python counterpart, say so in the PR body rather than forcing it.
