from __future__ import annotations

import re
from html import escape
from typing import Any


MD_TABLE_RE = re.compile(r"(?:^|\n)(\|.+\|\n\|[-:| ]+\|\n(?:\|.*\|\n?)*)", re.M)


def markdown_table_to_html(md: str) -> str:
    match = MD_TABLE_RE.search(md or "")
    if not match:
        return ""
    table_text = match.group(1).strip()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return ""
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    header = rows[0]
    body = rows[2:]
    thead = "<thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr></thead>"
    tbody_rows = []
    for row in body:
        tbody_rows.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def extract_best_table_html(raw_obj: Any) -> str:
    candidates: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lowered = str(key).lower()
                if lowered in {"html", "table_html", "res_html"} and isinstance(value, str) and "<table" in value:
                    candidates.append(value)
                if lowered in {"markdown", "md"} and isinstance(value, str):
                    html = markdown_table_to_html(value)
                    if html:
                        candidates.append(html)
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(raw_obj)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return ""
