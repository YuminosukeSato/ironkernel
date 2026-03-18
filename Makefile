.PHONY: all lint lint-rust lint-python fmt fmt-rust fmt-python \
	test test-rust test-python typecheck build dev clean

all: lint test typecheck

lint: lint-rust lint-python

lint-rust:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

lint-python:
	uv run ruff check python/ tests/
	uv run ruff format --check python/ tests/

fmt: fmt-rust fmt-python

fmt-rust:
	cargo fmt

fmt-python:
	uv run ruff format python/ tests/
	uv run ruff check --fix python/ tests/

test: test-rust test-python

test-rust:
	cargo test

test-python: dev
	uv run pytest tests/python/ -v

typecheck: dev
	uv run mypy python/ironkernel/ --strict

build:
	uv run maturin build --release

dev:
	uv run maturin develop

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info/
