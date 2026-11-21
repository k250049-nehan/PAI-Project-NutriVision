"""Command-line health checks. All logic lives in app.py."""

from pathlib import Path

from app import run_checks


report = run_checks()
Path("evaluation_report.md").write_text(report, encoding="utf-8")
print(report)
