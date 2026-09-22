# Build Jobs Review

*Review ClearML build jobs for failure handling, resource limits, and the
contract with the caller that consumes their output.*

Governs `machine/jobs/**/*.py`.

This code runs unattended against remote services, so review it for what happens
when something else fails, not for the successful build.

- A job that is cancelled, preempted, or killed must end without raising
  (`daa1b8e`). Check that cancellation is observed at every await or long loop,
  not only at the top.
- Memory is a real limit here (`4faa596`). For a change that grows a batch, a
  cache, or an in-memory corpus, say what bounds it.
- Remote storage and ClearML calls fail transiently (`de29377`). A retry must be
  bounded and an unrecoverable error must say which artifact and which step.
- Output is a contract. Pretranslations, alignments, and their indices are
  consumed by Serval, so a change to their shape or ordering is a breaking
  change even though nothing in this repo reads them (`0354cc7`).
- This subpackage needs the `jobs` extra and is not installed by a plain
  `pip install sil-machine`. A new import must not leak into the base package.
- Model and trainer changes belong with a measured artifact, not an assertion
  (`50cb1de`).
