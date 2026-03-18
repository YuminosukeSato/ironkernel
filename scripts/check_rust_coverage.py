#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


ZERO_LINE_RE = re.compile(r"^\s*(\d+)\|\s*0\|(.*)$")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "coverage-rust.txt"
    current_file: Path | None = None
    failures: list[str] = []
    source_cache: dict[Path, list[str]] = {}

    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        if raw_line.endswith(":") and raw_line.startswith(str(root)):
            current_file = Path(raw_line[:-1])
            continue

        match = ZERO_LINE_RE.match(raw_line)
        if match is None or current_file is None:
            continue
        if not current_file.is_relative_to(root / "src"):
            continue

        line_number = int(match.group(1))
        if current_file not in source_cache:
            source_cache[current_file] = current_file.read_text(encoding="utf-8").splitlines()
        source_line = source_cache[current_file][line_number - 1].strip()
        if source_line == "#[pymethods]":
            continue
        failures.append(f"{current_file}:{line_number}: {source_line}")

    if failures:
        print("Rust coverage gate failed. Unexpected uncovered lines:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Rust coverage gate passed for src/ (allowing only PyO3 #[pymethods] false positives).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
