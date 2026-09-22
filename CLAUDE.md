# machine.py

Much of this library is a Python port of [sillsdev/machine](https://github.com/sillsdev/machine) (C#).
When changing code that exists on both sides, check what the C# implementation does first — divergence
should be deliberate, not incidental.

## Comments

Follow the convention the surrounding code already establishes.

A comment that no longer describes the code below it is a defect. When changing logic, update or
delete the comments that explain it.

## Breaking changes

The names a subpackage re-exports are what downstream packages import from the published wheel, so
changing an exported signature or its semantics is a breaking change even when every caller inside
this repo still works. Say so in the PR description.

Watch for changes that fail silently rather than loudly — a parameter widened from `dict` to an
iterable of dicts will accept a positional `dict` and iterate its keys.

## Reviewing changes

CI covers lint, type and test breakage on every push, so re-running the suite during review adds
nothing. Use the environment for what CI can't do: check whether a suspected bug actually manifests.
Write a throwaway script or run one targeted test file, and compare against the base branch when the
question is whether behavior changed. A finding you tried and failed to reproduce is worth more than
one you only reasoned about.
