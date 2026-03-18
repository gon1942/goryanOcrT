from __future__ import annotations

import re
from html import unescape

from .schemas import OCRBlock, OCRDocument
from .table_rebuilder import markdown_table_to_html
from .utils.text import safe_html


STYLE = """
body{font-family:Arial,\"Apple SD Gothic Neo\",\"Noto Sans KR\",sans-serif;margin:24px;line-height:1.5;color:#222}
.page{margin:0 0 32px 0;padding:20px;border:1px solid #ddd;border-radius:12px}
.page-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#f3f4f6;font-size:12px}
.badge.warn{background:#fff3cd;color:#8a6d3b}
.block{margin:10px 0}
.block.table{overflow:auto}
.block .meta{font-size:12px;color:#666;margin-bottom:6px}
table{border-collapse:collapse;width:100%;margin:12px 0}
th,td{border:1px solid #444;padding:6px 8px;vertical-align:top}
th{background:#f5f5f5}
pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:8px}
"""

HTML_FRAGMENT_RE = re.compile(r"<\s*(table|div|p|h[1-6]|ul|ol|li|blockquote|span|section|article)\b", re.I)


def _looks_like_html_fragment(text: str) -> bool:
    candidate = unescape((text or "").strip())
    if not candidate:
        return False
    return bool(HTML_FRAGMENT_RE.search(candidate))


def _extract_renderable_html(block: OCRBlock) -> str:
    # 1) explicit HTML from OCR block
    if block.html and _looks_like_html_fragment(block.html):
        return unescape(block.html)

    # 2) text field may contain escaped HTML fragments from OCR-VL fallback
    if block.text and _looks_like_html_fragment(block.text):
        return unescape(block.text)

    # 3) markdown field may already contain raw HTML or a markdown table
    if block.markdown:
        if _looks_like_html_fragment(block.markdown):
            return unescape(block.markdown)
        md_html = markdown_table_to_html(block.markdown)
        if md_html:
            return md_html

    # 4) table blocks sometimes carry HTML-like text inside extra payloads
    for key in ("html", "table_html", "render_html", "raw_html"):
        value = block.extra.get(key) if isinstance(block.extra, dict) else None
        if isinstance(value, str) and _looks_like_html_fragment(value):
            return unescape(value)

    return ""


def _render_block_content(block: OCRBlock) -> str:
    html_fragment = _extract_renderable_html(block)
    if html_fragment:
        return html_fragment

    if block.markdown and block.block_type in {"table", "formula"}:
        return f"<pre>{safe_html(block.markdown)}</pre>"

    return f"<p>{safe_html(block.text)}</p>"


def render_document_html(doc: OCRDocument) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{safe_html(doc.source_file)}</title>",
        f"<style>{STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{safe_html(doc.source_file)}</h1>",
        f"<p>engine: {safe_html(doc.engine_used)} / overall_confidence: {doc.overall_confidence:.3f}</p>",
    ]
    for page in doc.pages:
        badge_class = "badge warn" if page.needs_review else "badge"
        parts.extend([
            f'<section class="page" id="page-{page.page_no}">',
            '<div class="page-header">',
            f"<h2>Page {page.page_no}</h2>",
            f'<span class="{badge_class}">confidence {page.page_confidence:.3f}</span>',
            "</div>",
        ])
        if page.review_reason:
            parts.append(f"<p><strong>review:</strong> {safe_html(', '.join(page.review_reason))}</p>")
        for block in page.blocks:
            parts.append(f'<div class="block {safe_html(block.block_type)}">')
            parts.append(
                f"<div class=\"meta\">{safe_html(block.block_type)} / conf={block.confidence:.3f} / bbox={safe_html(str(block.bbox))}</div>"
            )
            parts.append(_render_block_content(block))
            parts.append("</div>")
        parts.append("</section>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)
