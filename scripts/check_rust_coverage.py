#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_LINE_PERCENT = 80.0
MIN_FUNCTION_PERCENT = 75.0


def summarize_src(report: dict[str, object], root: Path) -> tuple[int, int, int, int]:
    lines_total = 0
    lines_covered = 0
    functions_total = 0
    functions_covered = 0

    for entry in report["data"]:
        for file_report in entry["files"]:
            file_path = Path(file_report["filename"])
            if not file_path.is_relative_to(root / "src"):
                continue

            line_summary = file_report["summary"]["lines"]
            function_summary = file_report["summary"]["functions"]
            lines_total += line_summary["count"]
            lines_covered += line_summary["covered"]
            functions_total += function_summary["count"]
            functions_covered += function_summary["covered"]

    return lines_total, lines_covered, functions_total, functions_covered


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "coverage-rust.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lines_total, lines_covered, functions_total, functions_covered = summarize_src(report, root)
    line_percent = (lines_covered / lines_total) * 100 if lines_total else 0.0
    function_percent = (functions_covered / functions_total) * 100 if functions_total else 0.0

    print(
        "Rust coverage summary:"
        f" lines={lines_covered}/{lines_total} ({line_percent:.2f}%)"
        f", functions={functions_covered}/{functions_total} ({function_percent:.2f}%)"
    )

    if line_percent < MIN_LINE_PERCENT or function_percent < MIN_FUNCTION_PERCENT:
        print(
            "Rust coverage gate failed:"
            f" required lines>={MIN_LINE_PERCENT:.2f}%"
            f" and functions>={MIN_FUNCTION_PERCENT:.2f}%"
        )
        return 1

    print(
        "Rust coverage gate passed:"
        f" lines>={MIN_LINE_PERCENT:.2f}%"
        f" and functions>={MIN_FUNCTION_PERCENT:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
