# machine.py contributor and agent guide

What an agent must do here, and where looking at the tree will mislead you.
Everything else - layout, dependencies, what a workflow runs - read from the
tree; it is accurate and this file would only rot.

If this file disagrees with `local_check.sh`, `.github/workflows/ci.yml`, or the
code you are changing, prefer the executable behavior and say so.

## Validation

Run `./local_check.sh` from the repository root: it installs, formats with black
and isort, lints with flake8, type-checks with pyright, and tests. Do not skip a
failing step or report success without fresh output.

Agents must also run `./local_check.sh --agent-strict`, which makes
`scripts/comment_hygiene.py` blocking over the lines the branch adds. The
standard it enforces is the `code-comments` skill. Do not drop the flag to get a
run through.

CI collects coverage and `local_check.sh` does not, so a local run is never
coverage-equivalent.

## Where the tree misleads

- `ci.yml` triggers on `push`, not `pull_request`. A green check on a pull
  request reflects the pushed head, not the merge result, and not a PR gate.
- The `Comment hygiene` check is advisory and never fails. Its green tick is not
  evidence that the strict scan passed.
- `poetry install --all-extras` does not install torch. That lives in the `gpu`
  Poetry group, which is a group and not an extra. `eflomal` installs only on
  Linux, so an alignment path can be exercised on one machine and skipped on
  another without saying so.
- pyright runs in `basic` mode. An annotation it accepts is not evidence the
  types are sound under `strict`.
- Tests import helpers as `testutils.*`, which resolves only because
  `tests/conftest.py` appends that directory to `sys.path`. The package has no
  `__init__.py` and is not importable from elsewhere.
- Three different things are called a reference here: a `ScriptureRef`, a
  versification-mapped verse location, and ordinary object identity. Say which.
  Likewise a "row" is a corpus row and not a USFM line; a "segment" is a
  `ScriptureRef` path component in corpora but a text span in tokenization; and
  "alignment" is word alignment unless you name another domain.

## Changing code

- Add or update focused tests with every behavior change. Keep fixtures
  deterministic and platform assumptions explicit.
- USFM marker and paragraph handling is the defect class that ships here most
  often - unclosed and implicitly closed markers (`29088f8`, `962ca36`), an
  element in an unexpected position (`9868016`), and spacing around end markers
  (`417c95f`). Cover the malformed and empty input with the fix.
- Reference and versification arithmetic recurs next (`29d4dbb`, `2a80929`,
  `7d85f16`). Assert over the affected book, chapter, and verse mapping rather
  than over a rendered string.
- In `punctuation_analysis`, index by text element and not by code unit, and
  assume the chapter or verse is missing or unparsable (`53992c8`, `ca37757`).
  This area's history is almost entirely crash fixes from live Paratext data.
- Close streams, models, and trainers according to their contracts, and be
  explicit about who owns a handle that is passed in.
- The names a subpackage re-exports through `__all__` are what downstream
  packages import from the published `sil-machine` wheel. Changing an exported
  signature or its semantics is a breaking change even when every caller inside
  this repo still works. Watch for a change that fails silently rather than
  loudly: a parameter widened from `dict` to an iterable of dicts accepts a
  positional `dict` and iterates its keys.
- Prefer an existing abstraction to a parallel one.

## Branch hygiene

Preserve unrelated changes and pre-existing untracked files. Never use
`reset --hard`, `checkout --`, broad deletion, or broad staging as a cleanup
shortcut. Local review notes go in `.review/`, which `.gitignore` excludes.

Much of this library is a port of [sillsdev/machine](https://github.com/sillsdev/machine)
(C#). When changing code that exists on both sides, check what the C#
implementation does first - divergence should be deliberate, not incidental, and
a porting change links its source pull request. A workflow files the porting
issue after merge; do not hand-file a duplicate.

## Agent guidance

`CLAUDE.md` imports this file. Claude workflows live under `.claude/skills/`.
Path-scoped review rules live under `docs/review/`; match the changed path,
first row wins:

| Path glob | Rules file |
| --- | --- |
| `machine/corpora/**/*.py` | `docs/review/corpora-usfm.md` |
| `machine/punctuation_analysis/**/*.py` | `docs/review/punctuation.md` |
| `machine/jobs/**/*.py` | `docs/review/jobs.md` |
| any other `machine/**/*.py` | `docs/review/machine-library.md` |
| `tests/**/*.py` | `docs/review/machine-tests.md` |

`docs/review/devils-advocate.md` is an optional adversarial pass for a high-risk
change. Add a nested `AGENTS.md` only when a subtree needs different rules.
