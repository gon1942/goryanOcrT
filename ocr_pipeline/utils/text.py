from __future__ import annotations

import re
from html import escape


WS_RE = re.compile(r"\s+")
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def normalize_whitespace(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip()


def extract_numeric_tokens(text: str) -> list[str]:
    return NUMERIC_RE.findall(text or "")


def safe_html(text: str) -> str:
    return escape(text or "").replace("\n", "<br />")
