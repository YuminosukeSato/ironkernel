SHELL := /bin/bash

STRESS_ITERATIONS ?= 200
RUST_TEST_THREADS ?= 1

.PHONY: all sync lint lint-rust lint-python fmt fmt-rust fmt-python \
	test test-rust test-python typecheck build dev dev-frozen \
	coverage-python coverage-rust stress mutate-core verify-all clean

all: verify-all

sync:
	uv sync --frozen --dev

lint: lint-rust lint-python

lint-rust:
	cargo fmt --check
	cargo clippy --locked --all-targets -- -D warnings

lint-python: sync
	uv run ruff check python/ tests/
	uv run ruff format --check python/ tests/

fmt: fmt-rust fmt-python

fmt-rust:
	cargo fmt

fmt-python:
	uv run ruff format python/ tests/
	uv run ruff check --fix python/ tests/

test: test-rust test-python

test-rust: sync
	bash scripts/with_venv_python.sh cargo test --locked --workspace -- --test-threads=$(RUST_TEST_THREADS)

test-python: dev-frozen
	uv run pytest tests/python/ -v

typecheck: dev-frozen
	uv run mypy python/ironkernel/ --strict

build:
	uv run maturin build --release

dev:
	uv run maturin develop

dev-frozen: sync
	uv run maturin develop

coverage-python: dev-frozen
	COVERAGE_FILE=.coverage-python uv run coverage run -m pytest tests/python/ -q
	COVERAGE_FILE=.coverage-python uv run coverage report -m
	COVERAGE_FILE=.coverage-python uv run coverage xml -o coverage-python.xml

coverage-rust: sync
	bash scripts/run_rust_coverage.sh

stress: dev-frozen
	bash scripts/stress.sh $(STRESS_ITERATIONS)

mutate-core: sync
	bash scripts/run_mutants.sh

verify-all: lint test typecheck coverage-python coverage-rust

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info/ coverage-python.xml coverage-rust.xml coverage-rust.json coverage-rust.txt
