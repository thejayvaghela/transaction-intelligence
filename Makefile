# Common commands. Per-phase targets (data, train, eval, serve, predict) are added as we build.

.PHONY: install test lint format data

install:        ## Create / refresh the virtual environment
	uv sync

data:           ## Regenerate the dataset deterministically from configs/data.yaml
	uv run python scripts/build_dataset.py

test:           ## Run the test suite
	uv run pytest

lint:           ## Lint with ruff
	uv run ruff check src tests

format:         ## Format with black + ruff import sort/fixes
	uv run black src tests
	uv run ruff check --fix src tests
