#!/bin/bash
poetry install
pip install isort pyright

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