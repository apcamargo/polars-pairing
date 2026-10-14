SHELL=/bin/bash

.venv:  ## Set up virtual environment with uv
	uv venv --python python3
	uv pip install -r requirements.txt

install: .venv  ## Build and install the plugin (debug)
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv  ## Build and install the plugin (release)
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv  ## Run formatters and linters
	cargo fmt --all && cargo clippy --all-features
	.venv/bin/python -m ruff check . --fix --exit-non-zero-on-fix
	.venv/bin/python -m ruff format polars_pairing tests
	.venv/bin/mypy polars_pairing tests

test: .venv  ## Run the test suite
	.venv/bin/python -m pytest tests
