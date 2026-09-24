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

poetry install

if [ "$agent_strict" = true ]; then
  echo "=================== comment hygiene ================="
  poetry run python scripts/comment_hygiene.py
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

echo "======================= black ======================"
poetry run black .
echo "======================= flake8 ======================"
poetry run flake8 .
echo "======================= isort ======================"
poetry run isort .
echo "======================= pyright ======================"
poetry run pyright
echo "======================= pytest ======================"
poetry run pytest
