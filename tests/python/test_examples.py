"""Executable example regression tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np


def load_example_module(path: Path) -> ModuleType:
    spec = spec_from_file_location("goal_story_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_goal_story_example_runs_why_repository_example_must_not_drift_from_public_api() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    example_path = repo_root / "examples" / "goal_story.py"
    assert example_path.exists()

    module = load_example_module(example_path)
    result = module.main()

    np.testing.assert_array_equal(result["mapped"], np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64))
    np.testing.assert_allclose(
        result["transformed"],
        np.sqrt(np.abs(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)))
        + np.sin(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)),
        rtol=1e-10,
    )
    assert result["total"] == 10.0
    assert result["average"] == 2.5
    np.testing.assert_array_equal(result["received"], np.array([42.0, 43.0], dtype=np.float64))
    assert result["selected_index"] == 0
    np.testing.assert_array_equal(result["selected_value"], np.array([99.0], dtype=np.float64))
    np.testing.assert_array_equal(result["relu"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
