"""Markdown report writer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_report(out_dir: Path, title: str, summary: dict, sections: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Details")
    lines.extend(sections)
    content = "\n".join(lines).strip() + "\n"
    (out_dir / "report.md").write_text(content)
