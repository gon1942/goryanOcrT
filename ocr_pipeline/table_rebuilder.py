from __future__ import annotations

import re
from html import escape, unescape
from typing import Any


MD_TABLE_RE = re.compile(r"(?:^|\n)(\|.+\|\n\|[-:| ]+\|\n(?:\|.*\|\n?)*)", re.M)
HTML_TABLE_RE = re.compile(r"(<table\b.*?</table>)", re.I | re.S)
HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ATTR_SPAN_RE = re.compile(r"""\b(?P<name>rowspan|colspan)\s*=\s*["']?(?P<value>\d+)["']?""", re.I)
_CLEAN_STYLE_RE = re.compile(r"""\s*style\s*=\s*["'][^"']*["']""", re.I)
_CLEAN_OPEN_TAG_RE = re.compile(r"<(t[dh])\b([^>]*)>", re.I)


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
                if lowered in {"html", "table_html", "res_html", "block_content"} and isinstance(value, str) and "<table" in value:
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


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to a markdown pipe table.

    Handles rowspan/colspan by expanding merged cells into the grid.
    """
    if not html:
        return ""
    table_match = HTML_TABLE_RE.search(html)
    if not table_match:
        return ""
    table_html = table_match.group(1)
    row_matches = HTML_ROW_RE.findall(table_html)
    if len(row_matches) < 2:
        return ""

    rows: list[list[tuple[str, int, int]]] = []
    for row_html in row_matches:
        cells: list[tuple[str, int, int]] = []
        for cell_match in HTML_CELL_RE.finditer(row_html):
            cell_html = cell_match.group(1)
            cell_text = unescape(HTML_TAG_RE.sub(" ", cell_html)).strip()
            attrs = cell_match.group(0)
            spans = {m.group("name").lower(): int(m.group("value")) for m in ATTR_SPAN_RE.finditer(attrs)}
            rs = spans.get("rowspan", 1)
            cs = spans.get("colspan", 1)
            cells.append((cell_text, rs, cs))
        rows.append(cells)

    if not rows:
        return ""

    max_cols = 0
    for row in rows:
        col_span = sum(cs for _, _, cs in row)
        if col_span > max_cols:
            max_cols = col_span
    if max_cols == 0:
        return ""

    grid: list[list[str]] = []
    for _ in rows:
        grid.append([""] * max_cols)

    occupied: list[list[bool]] = []
    for _ in rows:
        occupied.append([False] * max_cols)

    for r_idx, row in enumerate(rows):
        c_idx = 0
        for text, rs, cs in row:
            while c_idx < max_cols and occupied[r_idx][c_idx]:
                c_idx += 1
            if c_idx >= max_cols:
                break
            for dr in range(rs):
                for dc in range(cs):
                    rr = r_idx + dr
                    cc = c_idx + dc
                    if rr < len(grid) and cc < max_cols:
                        grid[rr][cc] = text if (dr == 0 and dc == 0) else ""
                        occupied[rr][cc] = True
            c_idx += cs

    md_lines: list[str] = []
    for i, row in enumerate(grid):
        line = "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        md_lines.append(line)
        if i == 0:
            md_lines.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(md_lines)


def clean_html_table(html: str) -> str:
    """Clean an HTML table for embedding in markdown.

    Keeps structural attributes (border, rowspan, colspan) but removes
    verbose inline styles and normalises whitespace.  The result renders
    correctly in any markdown viewer that supports HTML (GitHub, Obsidian,
    VS Code, etc.) while preserving rowspan/colspan that pipe tables cannot.
    """
    if not html:
        return ""
    table_match = HTML_TABLE_RE.search(html)
    if not table_match:
        return ""
    table_html = table_match.group(1)

    # Strip inline styles from the <table> open tag
    table_html = _CLEAN_STYLE_RE.sub("", table_html)

    # Strip inline styles from every <td>/<th> open tag, keep spans
    def _clean_cell(m: re.Match) -> str:
        tag = m.group(1)
        attrs = m.group(2)
        attrs = _CLEAN_STYLE_RE.sub("", attrs)
        return f"<{tag}{attrs}>"

    table_html = _CLEAN_OPEN_TAG_RE.sub(_clean_cell, table_html)

    # Normalise whitespace between tags
    table_html = re.sub(r">\s+<", "><", table_html)

    return table_html.strip()


# ---------------------------------------------------------------------------
# Internal helpers for repair_html_table_spans
# ---------------------------------------------------------------------------
_CELL_OPEN_RE = re.compile(r"<(t[dh])(\b[^>]*)>", re.I)
_CELL_CLOSE_RE = re.compile(r"</t[dh]>", re.I)
_ROW_RE = re.compile(r"(<tr\b[^>]*>)(.*?)(</tr>)", re.I | re.S)


def repair_html_table_spans(html: str) -> str:
    """Post-process PaddleVL HTML to fix structural issues.

    Fixes:
    1. Rowspan/colspan overflow — cells that would extend past the last row.
    2. Trailing empty cells in rows that are shorter than max_cols (artifact
       from PaddleVL adding placeholder ``<td></td>`` to pad colspan).
    3. Orphaned empty trailing columns that appear only in a few rows.
    """
    if not html:
        return ""
    table_match = HTML_TABLE_RE.search(html)
    if not table_match:
        return html

    open_tag, inner, close_tag = table_match.group(1).split("</table>", 1)[0], "", ""
    # Re-extract properly
    m = re.search(r"(<table\b[^>]*>)(.*?)(</table>)", html, re.I | re.S)
    if not m:
        return html
    open_tag, inner_html, close_tag = m.group(1), m.group(2), m.group(3)

    rows = _ROW_RE.findall(inner_html)
    if len(rows) < 2:
        return html

    # Parse rows into structured form: list of list of (tag, attrs_str, inner_text, full_match)
    parsed_rows: list[list[dict]] = []
    for _, row_inner, _ in rows:
        cells: list[dict] = []
        for cm in _CELL_OPEN_RE.finditer(row_inner):
            tag = cm.group(1)
            attrs = cm.group(2)
            # Find matching close tag
            after_open = row_inner[cm.end():]
            close_m = _CELL_CLOSE_RE.search(after_open)
            if close_m:
                cell_inner = after_open[:close_m.start()]
            else:
                cell_inner = ""
            text = unescape(HTML_TAG_RE.sub(" ", cell_inner)).strip()
            spans = {m.group("name").lower(): int(m.group("value")) for m in ATTR_SPAN_RE.finditer(attrs)}
            rs = int(spans.get("rowspan", 1))
            cs = int(spans.get("colspan", 1))
            cells.append({
                "tag": tag,
                "attrs": attrs,
                "text": text,
                "rs": rs,
                "cs": cs,
                "full": cm.group(0) + cell_inner + ("</" + tag + ">"),
            })
        parsed_rows.append(cells)

    num_rows = len(parsed_rows)

    # --- Fix 1: Rowspan overflow ---
    for ri, row in enumerate(parsed_rows):
        for cell in row:
            if cell["rs"] > 1 and ri + cell["rs"] > num_rows:
                cell["rs"] = num_rows - ri
                cell["attrs"] = _set_span_attr(cell["attrs"], "rowspan", cell["rs"])
                cell["full"] = _rebuild_cell(cell)

    # --- Fix 2: Trailing empty cells cleanup ---
    # Find the maximum column that has content in any row
    max_used_col = 0
    for row in parsed_rows:
        col = 0
        for cell in row:
            if cell["text"] and col + cell["cs"] > max_used_col:
                max_used_col = col + cell["cs"]
            col += cell["cs"]

    # Remove trailing empty cells that start at or beyond max_used_col
    for row in parsed_rows:
        while row:
            cell = row[-1]
            if cell["text"]:
                break
            col_start = sum(c["cs"] for c in row[:-1])
            if col_start >= max_used_col:
                row.pop()
            else:
                break

    # --- Rebuild HTML ---
    new_rows: list[str] = []
    for row in parsed_rows:
        row_html = "".join(c["full"] for c in row)
        new_rows.append(f"<tr>{row_html}</tr>")

    return f"{open_tag}{''.join(new_rows)}{close_tag}"


def _set_span_attr(attrs: str, name: str, value: int) -> str:
    """Set or remove a rowspan/colspan attribute."""
    if value <= 1:
        return re.sub(rf"""\s*\b{name}\s*=\s*["']?\d+["']?""", "", attrs, flags=re.I)
    if re.search(rf"\b{name}\s*=", attrs, flags=re.I):
        return re.sub(
            rf"""(\b{name}\s*=\s*["']?)\d+(["']?)""",
            rf"\g<1>{value}\2", attrs, flags=re.I,
        )
    return f'{attrs} {name}="{value}"'


def _rebuild_cell(cell: dict) -> str:
    """Rebuild a cell's full HTML from its parts."""
    escaped = escape(cell["text"]) if cell["text"] else ""
    return f"<{cell['tag']}{cell['attrs']}>{escaped}</{cell['tag']}>"
