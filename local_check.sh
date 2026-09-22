#!/bin/bash
uv sync --all-extras

echo "======================= black ======================"
uv run black .
echo "======================= flake8 ======================"
uv run flake8 .
echo "======================= isort ======================"
uv run isort .
echo "======================= pyright ======================"
uv run pyright
echo "======================= pytest ======================"
uv run pytest