#!/bin/bash
poetry install
pip install isort pyright
poetry run black .
poetry run flake8 .
poetry run isort .
poetry run pyright
poetry run pytest