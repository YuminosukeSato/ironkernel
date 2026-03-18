from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cargo_toml_keeps_extension_builds_free_of_auto_initialize_why_manylinux_wheels_must_not_embed_python() -> None:
    cargo_toml = read("Cargo.toml")

    assert 'pyo3 = "0.23"' in cargo_toml
    assert "auto-initialize" not in cargo_toml


def test_rust_entrypoints_bind_to_project_virtualenv_why_numpy_backed_rust_tests_need_the_same_python_as_pytest() -> (
    None
):
    makefile = read("Makefile")
    stress = read("scripts/stress.sh")
    coverage = read("scripts/run_rust_coverage.sh")

    assert "bash scripts/with_venv_python.sh cargo test --locked --workspace" in makefile
    assert "bash scripts/with_venv_python.sh cargo test --locked -q" in stress
    assert 'export PYO3_PYTHON="${VENV_PYTHON}"' in coverage
    assert 'export PYTHON_SYS_EXECUTABLE="${VENV_PYTHON}"' in coverage
    assert "coverage-rust.json" in coverage


def test_release_workflow_pins_linux_interpreters_why_manylinux_builds_must_target_supported_versions() -> None:
    workflow = read(".github/workflows/release.yml")

    assert "-i python3.9 python3.10 python3.11 python3.12 python3.13" in workflow
    assert 'manylinux: "2014"' in workflow
    assert "- os: ubuntu-latest\n            target: aarch64" not in workflow
