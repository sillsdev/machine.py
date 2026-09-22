# Machine Library Review

*Review shipped library code for public API compatibility, deterministic
comparison, and resource ownership.*

Governs any `machine/**/*.py` no more specific rules file claims.

- Check the shape of a changed exported signature, its defaults, and its return
  type; an in-repo clean rename is still a breaking change for a downstream
  caller (`deb112b`).
- Watch for a change that fails silently rather than loudly. Widening a
  parameter from a mapping to an iterable of mappings still accepts the old
  positional argument and iterates its keys; a keyword-only parameter, or a
  different name, fails at the call site instead.
- Compare markers, tokens, identifiers, and protocol text by exact string
  identity. Reserve case folding and locale-aware collation for genuinely
  linguistic operations.
- Close streams, models, engines, and trainers according to their contracts.
  Prefer a context manager to a `close` the caller must remember, and state who
  owns a handle that is passed in.
- Review a changed annotation for a false promise; do not ask for a
  repository-wide typing migration.
- Code reachable from a plain install must not import an optional extra at
  module scope.
- Add focused tests for the decisions the diff changes. Report the commands you
  ran; a coverage percentage is not evidence.
