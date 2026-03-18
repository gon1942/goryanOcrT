from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from html import escape, unescape
from pathlib import Path
import re
from typing import Any

from .pdf_utils import extract_pdf_page_texts, render_pdf_to_images
from .schemas import OCRBlock, OCRDocument, OCRPage
from .utils.io import ensure_dir
from .utils.text import normalize_whitespace, safe_html


TAG_RE = re.compile(r"<[^>]+>")
BLOCK_TAG_RE = re.compile(r"</?(?:tr|p|div|section|article|h[1-6]|li|ul|ol|blockquote|br)\b[^>]*>", re.I)
CELL_TAG_RE = re.compile(r"</?(?:td|th)\b[^>]*>", re.I)
HTML_FRAGMENT_RE = re.compile(r"<\s*(table|div|p|h[1-6]|ul|ol|li|blockquote|span|section|article)\b", re.I)


@dataclass(slots=True)
class BlockComparison:
    block_id: str
    block_type: str
    page_no: int
    status: str
    similarity: float
    source_text: str = ""
    ocr_text: str = ""
    review_reason: list[str] = field(default_factory=list)
    bbox: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageComparison:
    page_no: int
    source_available: bool
    source_kind: str
    source_text: str = ""
    ocr_text: str = ""
    similarity: float = 0.0
    status: str = "not_compared"
    source_image: str = ""
    text_diff_html: str = ""
    mismatched_blocks: list[BlockComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "source_available": self.source_available,
            "source_kind": self.source_kind,
            "source_text": self.source_text,
            "ocr_text": self.ocr_text,
            "similarity": self.similarity,
            "status": self.status,
            "source_image": self.source_image,
            "text_diff_html": self.text_diff_html,
            "mismatched_blocks": [block.to_dict() for block in self.mismatched_blocks],
        }


@dataclass(slots=True)
class DocumentComparison:
    source_file: str
    comparable: bool
    source_kind: str
    overall_similarity: float
    summary_status: str
    pages: list[PageComparison]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "comparable": self.comparable,
            "source_kind": self.source_kind,
            "overall_similarity": self.overall_similarity,
            "summary_status": self.summary_status,
            "notes": self.notes,
            "pages": [page.to_dict() for page in self.pages],
        }


def build_document_comparison(input_path: str | Path, doc: OCRDocument, output_dir: str | Path) -> DocumentComparison:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    compare_dir = ensure_dir(output_dir / "comparison")

    if input_path.suffix.lower() == ".pdf":
        source_images = render_pdf_to_images(input_path, compare_dir / "source_pages")
    else:
        source_images = [input_path]

    if input_path.suffix.lower() == ".pdf" and doc.metadata.get("profile", {}).get("has_text_layer"):
        return _compare_text_layer_pdf(input_path, doc, source_images)
    return _compare_visual_only(input_path, doc, source_images)


def render_comparison_html(comparison: DocumentComparison, doc: OCRDocument | None = None) -> str:
    page_map = {page.page_no: page for page in (doc.pages if doc else [])}
    style = """
body{font-family:Arial,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;margin:24px;background:#f6f2ea;color:#1e1e1e}
h1,h2,h3{margin:0 0 12px}
.summary{margin-bottom:24px;padding:16px 18px;background:#fff;border:1px solid #d8d1c2;border-radius:14px}
.page{margin-bottom:24px;padding:18px;background:#fff;border:1px solid #d8d1c2;border-radius:14px}
.meta{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#ece6d8;font-size:12px}
.badge.warn{background:#ffd9b8}
.badge.bad{background:#ffb7b7}
.grid{display:grid;grid-template-columns:1.1fr 1fr;gap:18px;align-items:start}
.panel{border:1px solid #e4dece;border-radius:12px;padding:14px;background:#fffdfa}
.panel img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:10px}
.diff{white-space:pre-wrap;line-height:1.6}
ins{background:#d7f5d1;text-decoration:none}
del{background:#ffd6d6;text-decoration:none}
.blocks{margin-top:16px}
.block{padding:10px 12px;border:1px solid #ece6d8;border-radius:10px;margin-bottom:10px;background:#fffdfa}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.hint{color:#5d5445;font-size:14px}
.text-preview{white-space:pre-wrap;line-height:1.55;max-height:260px;overflow:auto;background:#f7f2e8;border:1px solid #e6decd;border-radius:10px;padding:12px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.page-compare{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.page-viewer{border:1px solid #e4dece;border-radius:12px;padding:14px;background:#fffdfa}
.page-viewer img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:10px;display:block}
.ocr-page{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;max-height:900px;overflow:auto}
.ocr-block{margin:0 0 12px;padding:10px 12px;border:1px solid #ece6d8;border-radius:10px;background:#fffdfa}
.ocr-block.diff{border-color:#d95f5f;background:#fff1f1;box-shadow:0 0 0 2px rgba(217,95,95,.12) inset}
.ocr-block .label{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:#6b6457;margin-bottom:8px}
.ocr-block table{width:100%;border-collapse:collapse}
.ocr-block th,.ocr-block td{border:1px solid #555;padding:4px 6px;vertical-align:top}
.ocr-block th{background:#f4efe6}
.ocr-block pre{white-space:pre-wrap;margin:0;background:#f6f1e7;padding:10px;border-radius:8px}
.legend{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 0}
"""
    parts = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{safe_html(comparison.source_file)} comparison</title>",
        f"<style>{style}</style>",
        "</head>",
        "<body>",
        '<section class="summary">',
        f"<h1>{safe_html(comparison.source_file)} 비교 결과</h1>",
        '<div class="meta">',
        _badge(f"source={comparison.source_kind}", "normal"),
        _badge(f"overall_similarity={comparison.overall_similarity:.3f}", _status_tone(comparison.summary_status)),
        _badge(comparison.summary_status, _status_tone(comparison.summary_status)),
        "</div>",
    ]
    for note in comparison.notes:
        parts.append(f"<p>{safe_html(note)}</p>")
    parts.append('<p class="hint">비교 기준: 원본 PDF에서 추출한 텍스트와 OCR 결과에서 추출한 표시용 텍스트를 페이지 단위로 비교합니다. 표 HTML은 비교 전에 텍스트로 변환합니다.</p>')
    parts.append('<div class="legend">')
    parts.append(_badge("왼쪽: 원본 페이지", "normal"))
    parts.append(_badge("오른쪽: OCR 결과 페이지", "normal"))
    parts.append(_badge("빨간 블록: 상이하거나 검토 필요한 OCR 블록", "bad"))
    parts.append('</div>')
    parts.append("</section>")

    for page in comparison.pages:
        ocr_page = page_map.get(page.page_no)
        mismatch_ids = {block.block_id for block in page.mismatched_blocks if block.block_id and not block.block_id.startswith("source_line_")}
        parts.extend([
            '<section class="page">',
            f"<h2>Page {page.page_no}</h2>",
            '<div class="meta">',
            _badge(page.source_kind, "normal"),
            _badge(f"similarity={page.similarity:.3f}", _status_tone(page.status)),
            _badge(page.status, _status_tone(page.status)),
            _badge(f"mismatches={len(page.mismatched_blocks)}", "warn" if page.mismatched_blocks else "normal"),
            "</div>",
            '<div class="stats">',
            _badge(f"source_chars={len(page.source_text)}", "normal"),
            _badge(f"ocr_chars={len(page.ocr_text)}", "normal"),
            '</div>',
            '<div class="page-compare">',
            '<div class="page-viewer">',
            "<h3>원본 페이지</h3>",
        ])
        if page.source_image:
            parts.append(f'<img src="{escape(page.source_image, quote=True)}" alt="source page {page.page_no}" />')
        else:
            parts.append("<p>원본 이미지 미사용</p>")
        parts.extend([
            "</div>",
            '<div class="page-viewer">',
            "<h3>결과 페이지</h3>",
            '<p class="hint">아래는 OCR 결과를 페이지 형태로 다시 렌더한 것입니다. 상이한 블록은 빨간 배경으로 표시됩니다.</p>',
        ])
        if ocr_page:
            parts.append(_render_result_page(ocr_page, mismatch_ids))
        else:
            parts.append("<p>결과 페이지 데이터를 찾을 수 없습니다.</p>")
        parts.extend(["</div>", "</div>"])
        parts.extend([
            '<div class="panel" style="margin-top:16px;">',
            "<h3>텍스트 차이</h3>",
        ])
        if page.text_diff_html:
            parts.append('<p class="hint">빨강은 원본에는 있는데 OCR 결과 텍스트에서 사라진 내용, 초록은 OCR 결과에만 있는 내용입니다.</p>')
            parts.append(f'<div class="diff">{page.text_diff_html}</div>')
        else:
            parts.append("<p>텍스트 레이어가 없어 diff 대신 검수 대상 블록만 표시합니다.</p>")
        parts.append("</div>")
        parts.extend([
            '<div class="grid" style="margin-top:16px;">',
            '<div class="panel">',
            "<h3>원본 추출 텍스트</h3>",
            f'<div class="text-preview">{safe_html(_truncate(page.source_text, 4000) or "(없음)")}</div>',
            "</div>",
            '<div class="panel">',
            "<h3>OCR 비교용 텍스트</h3>",
            f'<div class="text-preview">{safe_html(_truncate(page.ocr_text, 4000) or "(없음)")}</div>',
            "</div>",
            "</div>",
        ])
        if page.mismatched_blocks:
            parts.append('<div class="blocks"><h3>확인 필요한 블록</h3>')
            for block in page.mismatched_blocks[:40]:
                parts.append('<div class="block">')
                parts.append(
                    f"<p><strong>{safe_html(block.block_id)}</strong> / {safe_html(block.block_type)} / "
                    f"status={safe_html(block.status)} / similarity={block.similarity:.3f}</p>"
                )
                if block.review_reason:
                    parts.append(f"<p>reason: {safe_html(', '.join(block.review_reason))}</p>")
                if block.source_text:
                    parts.append(f"<p><strong>source</strong>: {safe_html(_truncate(block.source_text))}</p>")
                if block.ocr_text:
                    parts.append(f"<p><strong>ocr</strong>: {safe_html(_truncate(block.ocr_text))}</p>")
                parts.append(f"<p class=\"mono\">bbox={safe_html(str(block.bbox))}</p>")
                parts.append("</div>")
            if len(page.mismatched_blocks) > 40:
                parts.append(f"<p class=\"hint\">표시를 위해 상위 40개만 노출했습니다. 전체 mismatch 수: {len(page.mismatched_blocks)}</p>")
            parts.append("</div>")
        parts.append("</section>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def _compare_text_layer_pdf(input_path: Path, doc: OCRDocument, source_images: list[Path]) -> DocumentComparison:
    source_texts = extract_pdf_page_texts(input_path)
    pages: list[PageComparison] = []
    for idx, page in enumerate(doc.pages, start=1):
        source_text = source_texts[idx - 1] if idx - 1 < len(source_texts) else ""
        ocr_text = _page_text(page)
        similarity = _similarity(source_text, ocr_text)
        status = _page_status(similarity, source_text, ocr_text)
        mismatched_blocks = _compare_blocks_with_source_text(
            page,
            source_text,
            include_source_only=similarity < 0.90,
        )
        if similarity >= 0.95:
            mismatched_blocks = []
        page_comparison = PageComparison(
            page_no=page.page_no,
            source_available=bool(source_text),
            source_kind="text_layer_pdf",
            source_text=source_text,
            ocr_text=ocr_text,
            similarity=similarity,
            status=status,
            source_image=_html_path(source_images[idx - 1]) if idx - 1 < len(source_images) else "",
            text_diff_html=_diff_html(source_text, ocr_text),
            mismatched_blocks=mismatched_blocks,
        )
        pages.append(page_comparison)
    overall = _mean([page.similarity for page in pages]) if pages else 0.0
    return DocumentComparison(
        source_file=doc.source_file,
        comparable=True,
        source_kind="text_layer_pdf",
        overall_similarity=overall,
        summary_status=_summary_status(overall, pages),
        pages=pages,
        notes=["텍스트 레이어 PDF는 원본 텍스트와 OCR 텍스트를 직접 비교했습니다."],
    )


def _compare_visual_only(input_path: Path, doc: OCRDocument, source_images: list[Path]) -> DocumentComparison:
    pages: list[PageComparison] = []
    notes = [
        "원본 텍스트 레이어가 없어 문자열 diff는 생략했습니다.",
        "대신 review 플래그 또는 낮은 confidence 블록을 검수 대상으로 표시합니다.",
    ]
    for idx, page in enumerate(doc.pages, start=1):
        mismatched = _collect_visual_review_blocks(page)
        status = "needs_review" if mismatched else "review_ok"
        similarity = 1.0 if not mismatched else max(0.0, 1.0 - min(0.8, len(mismatched) * 0.1))
        pages.append(
            PageComparison(
                page_no=page.page_no,
                source_available=False,
                source_kind="image_or_scanned",
                source_text="",
                ocr_text=_page_text(page),
                similarity=similarity,
                status=status,
                source_image=_html_path(source_images[idx - 1]) if idx - 1 < len(source_images) else "",
                text_diff_html="",
                mismatched_blocks=mismatched,
            )
        )
    overall = _mean([page.similarity for page in pages]) if pages else 0.0
    return DocumentComparison(
        source_file=doc.source_file,
        comparable=False,
        source_kind="image_or_scanned",
        overall_similarity=overall,
        summary_status=_summary_status(overall, pages),
        pages=pages,
        notes=notes,
    )


def _compare_blocks_with_source_text(page: OCRPage, source_text: str, *, include_source_only: bool = True) -> list[BlockComparison]:
    source_lines = [line for line in _source_units(source_text) if line]
    source_cursor = 0
    matched_indices: set[int] = set()
    results: list[BlockComparison] = []
    text_blocks = [block for block in page.blocks if _block_text(block)]

    for block in text_blocks:
        block_text = _block_text(block)
        best_idx, best_score = _best_match(source_lines, block_text, start=source_cursor)
        matched_source = source_lines[best_idx] if best_idx is not None else ""
        status = _block_status(best_score, matched_source, block_text)
        if best_idx is not None:
            source_cursor = best_idx + 1
            matched_indices.add(best_idx)
        if status != "match" or block.review:
            reasons = list(block.review_reason)
            if status != "match":
                reasons.append("text_diff")
            results.append(
                BlockComparison(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page_no=page.page_no,
                    status=status,
                    similarity=best_score,
                    source_text=matched_source,
                    ocr_text=block_text,
                    review_reason=sorted(set(reasons)),
                    bbox=block.bbox,
                )
            )

    if include_source_only:
        for idx, line in enumerate(source_lines):
            if idx in matched_indices:
                continue
            best_score = max((_similarity(line, _block_text(block)) for block in text_blocks), default=0.0)
            if best_score < 0.55:
                results.append(
                    BlockComparison(
                        block_id=f"source_line_{len(results) + 1}",
                        block_type="source_only",
                        page_no=page.page_no,
                        status="missing_in_ocr",
                        similarity=best_score,
                        source_text=line,
                        ocr_text="",
                        review_reason=["missing_in_ocr"],
                        bbox=[],
                    )
                )
    results.sort(key=lambda item: (item.status, item.block_id))
    return results


def _collect_visual_review_blocks(page: OCRPage) -> list[BlockComparison]:
    items: list[BlockComparison] = []
    for block in page.blocks:
        if block.review or block.confidence < 0.75:
            reasons = list(block.review_reason)
            if block.confidence < 0.75:
                reasons.append("low_confidence")
            items.append(
                BlockComparison(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page_no=page.page_no,
                    status="needs_review",
                    similarity=max(0.0, min(1.0, block.confidence)),
                    source_text="",
                    ocr_text=_block_text(block),
                    review_reason=sorted(set(reasons)),
                    bbox=block.bbox,
                )
            )
    return items


def _source_units(text: str) -> list[str]:
    lines = [normalize_whitespace(line) for line in (text or "").splitlines()]
    return [line for line in lines if line]


def _page_text(page: OCRPage) -> str:
    chunks: list[str] = []
    for block in page.blocks:
        text = _block_text(block)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _block_text(block: OCRBlock) -> str:
    if block.text:
        return normalize_whitespace(_html_to_text(block.text))
    if block.markdown:
        return normalize_whitespace(_html_to_text(block.markdown))
    if block.html:
        return normalize_whitespace(_html_to_text(block.html))
    return ""


def _best_match(source_lines: list[str], target: str, start: int = 0) -> tuple[int | None, float]:
    best_idx = None
    best_score = 0.0
    if not target:
        return None, 0.0
    window_end = min(len(source_lines), start + 8) if start < len(source_lines) else len(source_lines)
    candidates = list(range(start, window_end)) + list(range(window_end, min(len(source_lines), window_end + 4)))
    if not candidates:
        candidates = list(range(len(source_lines)))
    for idx in candidates:
        score = _similarity(source_lines[idx], target)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def _similarity(a: str, b: str) -> float:
    a_norm = normalize_whitespace(a)
    b_norm = normalize_whitespace(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _diff_html(source: str, ocr: str) -> str:
    source_tokens = (source or "").split()
    ocr_tokens = (ocr or "").split()
    matcher = SequenceMatcher(None, source_tokens, ocr_tokens)
    pieces: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pieces.append(escape(" ".join(source_tokens[i1:i2])))
        elif tag == "delete":
            pieces.append(f"<del>{escape(' '.join(source_tokens[i1:i2]))}</del>")
        elif tag == "insert":
            pieces.append(f"<ins>{escape(' '.join(ocr_tokens[j1:j2]))}</ins>")
        else:
            pieces.append(f"<del>{escape(' '.join(source_tokens[i1:i2]))}</del>")
            pieces.append(f"<ins>{escape(' '.join(ocr_tokens[j1:j2]))}</ins>")
    return " ".join(piece for piece in pieces if piece).strip()


def _page_status(similarity: float, source_text: str, ocr_text: str) -> str:
    if not source_text and not ocr_text:
        return "empty"
    if similarity >= 0.97:
        return "match"
    if similarity >= 0.85:
        return "partial"
    return "mismatch"


def _block_status(similarity: float, source_text: str, ocr_text: str) -> str:
    if source_text and not ocr_text:
        return "missing_in_ocr"
    if ocr_text and not source_text:
        return "extra_in_ocr"
    if similarity >= 0.97:
        return "match"
    if similarity >= 0.85:
        return "partial"
    return "mismatch"


def _summary_status(overall: float, pages: list[PageComparison]) -> str:
    if any(page.status in {"mismatch", "needs_review"} for page in pages):
        if overall < 0.85:
            return "attention_required"
    if overall >= 0.97:
        return "match"
    if overall >= 0.85:
        return "partial"
    return "attention_required"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _truncate(text: str, limit: int = 220) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _html_to_text(value: str) -> str:
    text = value or ""
    if "<" not in text or ">" not in text:
        return text
    text = BLOCK_TAG_RE.sub("\n", text)
    text = CELL_TAG_RE.sub("\t", text)
    text = TAG_RE.sub(" ", text)
    text = unescape(text)
    text = text.replace("\t\t", "\t")
    lines = []
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line.replace("\t", " "))
        if line:
            lines.append(line)
    return "\n".join(lines)


def _render_result_page(page: OCRPage, mismatch_ids: set[str]) -> str:
    parts = ['<div class="ocr-page">']
    for block in page.blocks:
        block_text = _block_render_html(block)
        classes = ["ocr-block"]
        if block.block_id in mismatch_ids or block.review:
            classes.append("diff")
        parts.append(f'<div class="{" ".join(classes)}" id="{safe_html(block.block_id)}">')
        parts.append('<div class="label">')
        parts.append(
            f"<span>{safe_html(block.block_type)} / {safe_html(block.block_id)}</span>"
            f"<span>conf={block.confidence:.3f}</span>"
        )
        parts.append("</div>")
        parts.append(block_text)
        parts.append("</div>")
    if not page.blocks:
        parts.append("<p>표시할 OCR 블록이 없습니다.</p>")
    parts.append("</div>")
    return "\n".join(parts)


def _block_render_html(block: OCRBlock) -> str:
    if block.html:
        return unescape(block.html)
    if block.text and _looks_like_html_fragment(block.text):
        return unescape(block.text)
    if block.markdown and _looks_like_html_fragment(block.markdown):
        return unescape(block.markdown)
    if block.markdown and "|" in block.markdown and "\n" in block.markdown:
        return f"<pre>{safe_html(block.markdown)}</pre>"
    if block.markdown and not block.text:
        return f"<pre>{safe_html(block.markdown)}</pre>"
    text = block.text or block.markdown or ""
    if block.block_type == "title":
        return f"<h4>{safe_html(text)}</h4>"
    return f"<div>{safe_html(text)}</div>"


def _looks_like_html_fragment(text: str) -> bool:
    candidate = unescape((text or "").strip())
    if not candidate:
        return False
    return bool(HTML_FRAGMENT_RE.search(candidate))


def _badge(text: str, tone: str) -> str:
    cls = "badge"
    if tone == "warn":
        cls += " warn"
    elif tone == "bad":
        cls += " bad"
    return f'<span class="{cls}">{safe_html(text)}</span>'


def _status_tone(status: str) -> str:
    if status in {"match", "review_ok"}:
        return "normal"
    if status in {"partial", "needs_review"}:
        return "warn"
    return "bad"


def _html_path(path: Path) -> str:
    return f"{path.parent.name}/{path.name}" if path.parent.name == "source_pages" else str(path)
