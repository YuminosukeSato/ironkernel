"""Stub package verification tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import ironkernel
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "python" / "ironkernel"


def _stub_all_exports(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                    raise AssertionError(f"__all__ in {path} is not a list literal")
    raise AssertionError(f"__all__ not found in {path}")


def test_py_typed_exists_why_typed_package_metadata_must_ship_with_the_release() -> None:
    assert (PACKAGE_DIR / "py.typed").is_file()


def test_all_public_types_in_stub_why_root_stub_exports_must_match_runtime_module_surface() -> None:
    for name in _stub_all_exports(PACKAGE_DIR / "__init__.pyi"):
        assert hasattr(ironkernel, name), name


def test_stub_signatures_match_runtime_why_public_stub_contracts_must_reflect_callable_shapes() -> None:
    chan_signature = inspect.signature(ironkernel.chan)
    assert tuple(chan_signature.parameters) == ("capacity",)

    go_signature = inspect.signature(ironkernel.rt.go)
    assert tuple(go_signature.parameters) == ("spec", "out")
    assert go_signature.parameters["out"].default is None

    where_signature = inspect.signature(ironkernel.kernel.where)
    assert tuple(where_signature.parameters) == ("cond", "true_val", "false_val")

    task = ironkernel.rt.go(ironkernel.kernel.sum(ironkernel.rt.asarray(np.array([1.0]))))
    try:
        result_signature = inspect.signature(task.result)
        assert tuple(result_signature.parameters) == ()
    finally:
        task.result()
