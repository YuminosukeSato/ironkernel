#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    outcomes_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mutants.out/outcomes.json")
    baseline_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("scripts/mutation-baseline.json")

    outcomes = load_json(outcomes_path)
    baseline = load_json(baseline_path)

    caught = int(outcomes.get("caught", 0))
    missed = int(outcomes.get("missed", 0))
    timeout = int(outcomes.get("timeout", 0))
    total_mutants = int(outcomes.get("total_mutants", 0))

    failures: list[str] = []
    if missed > int(baseline["max_missed"]):
        failures.append(f"missed mutants increased to {missed}")
    if timeout > int(baseline["max_timeout"]):
        failures.append(f"timed out mutants increased to {timeout}")
    if caught < int(baseline["min_caught"]):
        failures.append(f"caught mutants dropped to {caught}")

    summary = (
        f"mutation summary: total={total_mutants} caught={caught} "
        f"missed={missed} timeout={timeout}"
    )
    print(summary)

    if failures:
        print("Mutation gate failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Mutation gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
