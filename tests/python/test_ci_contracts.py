from __future__ import annotations

import os
import subprocess
import sysconfig
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


def test_rust_entrypoint_wrapper_exports_venv_site_packages_why_embedded_numpy_requires_project_site_packages() -> None:
    expected_python = ROOT / ".venv" / "bin" / "python"
    env_output = subprocess.run(
        ["bash", "scripts/with_venv_python.sh", "env"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported = dict(line.split("=", 1) for line in env_output.splitlines() if "=" in line)

    assert exported["PYO3_PYTHON"] == str(expected_python)
    assert exported["PYTHON_SYS_EXECUTABLE"] == str(expected_python)
    assert exported["VIRTUAL_ENV"] == str(ROOT / ".venv")
    assert exported["PATH"].split(os.pathsep)[0] == str(ROOT / ".venv" / "bin")
    assert sysconfig.get_path("purelib") in exported["PYTHONPATH"].split(os.pathsep)


def test_release_workflow_pins_linux_interpreters_why_manylinux_builds_must_target_supported_versions() -> None:
    workflow = read(".github/workflows/release.yml")

    assert "-i python3.9 python3.10 python3.11 python3.12 python3.13" in workflow
    assert 'manylinux: "2014"' in workflow
    assert "- os: ubuntu-latest\n            target: aarch64" not in workflow
    assert "- os: macos-13" not in workflow
