#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

MIN_LINE_PERCENT = 100.0
COUNTED_LINE_RE = re.compile(r"^\s*(\d+)\|\s*([^|]*)\|(.*)$")


def is_ignored_false_positive(source_text: str) -> bool:
    return source_text.strip() == "#[pymethods]"


def summarize_src_text(report_text: str, root: Path) -> tuple[int, int, list[tuple[Path, int]]]:
    covered = 0
    total = 0
    ignored: list[tuple[Path, int]] = []
    current_file: Path | None = None

    for raw_line in report_text.splitlines():
        if raw_line.endswith(":") and not raw_line.startswith(" "):
            file_path = Path(raw_line[:-1])
            current_file = file_path if file_path.is_relative_to(root / "src") else None
            continue

        if current_file is None:
            continue

        match = COUNTED_LINE_RE.match(raw_line)
        if match is None:
            continue

        line_number = int(match.group(1))
        count_text = match.group(2).strip()
        source_text = match.group(3)
        if not count_text:
            continue

        if count_text == "0" and is_ignored_false_positive(source_text):
            ignored.append((current_file, line_number))
            continue

        total += 1
        if count_text != "0":
            covered += 1

    return covered, total, ignored


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "coverage-rust.txt"
    covered, total, ignored = summarize_src_text(report_path.read_text(encoding="utf-8"), root)
    line_percent = (covered / total) * 100 if total else 100.0

    print(f"Rust coverage summary: lines={covered}/{total} ({line_percent:.2f}%)")

    if ignored:
        print("Ignored LLVM false positives:")
        for file_path, line_number in ignored:
            print(f" - {file_path}:{line_number}")

    if line_percent < MIN_LINE_PERCENT:
        print("Rust coverage gate failed:" f" required lines>={MIN_LINE_PERCENT:.2f}%")
        return 1

    print("Rust coverage gate passed:" f" lines>={MIN_LINE_PERCENT:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
