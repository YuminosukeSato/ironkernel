from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK_RUST_COVERAGE_PATH = ROOT / "scripts" / "check_rust_coverage.py"


def load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_rust_entrypoint_wrapper_is_passthrough_why_pyo3_uses_build_time_python() -> None:
    """with_venv_python.sh must be a thin passthrough (exec "$@").

    PyO3 links against the Python detected at cargo build time, so
    env-var overrides like PYO3_PYTHON do not change the interpreter
    at test runtime.  CI installs numpy via pip into the system Python.
    """
    wrapper = read("scripts/with_venv_python.sh")

    assert 'exec "$@"' in wrapper
    # Must NOT set PYTHONPATH — it causes numpy source-directory import errors.
    assert "PYTHONPATH" not in wrapper or "unset PYTHONPATH" in wrapper


def test_release_workflow_pins_linux_interpreters_why_manylinux_builds_must_target_supported_versions() -> None:
    workflow = read(".github/workflows/release.yml")

    assert "-i python3.9 python3.10 python3.11 python3.12 python3.13" in workflow
    assert 'manylinux: "2014"' in workflow
    assert "- os: ubuntu-latest\n            target: aarch64" not in workflow
    assert "- os: macos-13" not in workflow


def test_rust_coverage_checker_ignores_only_pymethods_lines_why_pyo3_annotations_are_false_positives(
    tmp_path: Path,
) -> None:
    checker = load_module(CHECK_RUST_COVERAGE_PATH, "check_rust_coverage")
    source_root = tmp_path / "project"
    src_dir = source_root / "src" / "python"
    src_dir.mkdir(parents=True)
    src_file = src_dir / "demo.rs"
    src_file.write_text(
        "\n".join(
            [
                "#[pymethods]",
                "fn covered() {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report_text = "\n".join(
        [
            f"{src_file}:",
            "    1|      0|#[pymethods]",
            "    2|      1|fn covered() {}",
            "",
        ]
    )

    covered, total, ignored = checker.summarize_src_text(report_text, source_root)

    assert (covered, total) == (1, 1)
    assert ignored == [(src_file, 1)]


def test_rust_coverage_checker_rejects_real_uncovered_lines_why_the_gate_must_fail_on_missed_runtime_logic(
    tmp_path: Path,
) -> None:
    checker = load_module(CHECK_RUST_COVERAGE_PATH, "check_rust_coverage")
    source_root = tmp_path / "project"
    src_dir = source_root / "src" / "runtime"
    src_dir.mkdir(parents=True)
    src_file = src_dir / "demo.rs"
    src_file.write_text(
        "\n".join(
            [
                "fn uncovered() {}",
                "fn covered() {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report_text = "\n".join(
        [
            f"{src_file}:",
            "    1|      0|fn uncovered() {}",
            "    2|      1|fn covered() {}",
            "",
        ]
    )

    covered, total, ignored = checker.summarize_src_text(report_text, source_root)

    assert (covered, total) == (1, 2)
    assert ignored == []
