#!/bin/bash
# Usage: ./local_check.sh [--agent-strict]
#
# --agent-strict adds the comment-hygiene check and fails on a violation in the
# lines this branch adds. Agents must pass it; humans need not.

agent_strict=false
if [ "${1:-}" = "--agent-strict" ]; then
  agent_strict=true
  shift
fi

if [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--agent-strict]" >&2
  exit 2
fi

uv sync --all-extras

if [ "$agent_strict" = true ]; then
  echo "=================== comment hygiene ================="
  uv run python scripts/comment_hygiene.py
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

# Every step runs even after one fails, and the last line names the failures, so
# output read through tail cannot pass off a pyright failure as a pytest pass.
failed=()

echo "======================= black ======================"
uv run black . || failed+=(black)
echo "======================= flake8 ======================"
uv run flake8 . || failed+=(flake8)
echo "======================= isort ======================"
uv run isort . || failed+=(isort)
echo "======================= pyright ======================"
uv run pyright || failed+=(pyright)
echo "======================= pytest ======================"
uv run pytest || failed+=(pytest)

if [ "${#failed[@]}" -ne 0 ]; then
  echo "local_check: FAILED (${failed[*]})"
  exit 1
fi
echo "local_check: PASSED"
