from __future__ import annotations

from statistics import mean

from .config import ReviewConfig
from .schemas import OCRBlock, OCRDocument, OCRPage
from .utils.text import extract_numeric_tokens


def score_block(block: OCRBlock, review: ReviewConfig) -> OCRBlock:
    score = block.confidence or 0.0
    reasons: list[str] = []
    if not (block.text or block.html or block.markdown):
        score -= 0.30
        reasons.append("empty_content")
    if block.block_type == "table" and not block.html:
        score -= review.empty_table_penalty
        reasons.append("table_without_html")
    if block.text:
        numeric_tokens = extract_numeric_tokens(block.text)
        if len(numeric_tokens) >= 3 and len(set(numeric_tokens)) == 1:
            score -= review.numeric_mismatch_penalty / 2
            reasons.append("repetitive_numeric_pattern")
    block.confidence = max(0.0, min(1.0, score))
    if block.confidence < review.block_score_threshold:
        block.review = True
        if "low_block_confidence" not in reasons:
            reasons.append("low_block_confidence")
    block.review_reason = sorted(set(block.review_reason + reasons))
    return block


def score_page(page: OCRPage, review: ReviewConfig) -> OCRPage:
    if page.blocks:
        page.page_confidence = mean([max(0.0, min(1.0, b.confidence)) for b in page.blocks])
    else:
        page.page_confidence = 0.0
        page.review_reason.append("page_without_blocks")
    if page.page_confidence < review.page_score_threshold or any(b.review for b in page.blocks):
        page.needs_review = True
        if page.page_confidence < review.page_score_threshold:
            page.review_reason.append("low_page_confidence")
    page.review_reason = sorted(set(page.review_reason))
    return page


def score_document(doc: OCRDocument, review: ReviewConfig) -> OCRDocument:
    for page in doc.pages:
        for block in page.blocks:
            score_block(block, review)
        score_page(page, review)
    meaningful_pages = [p for p in doc.pages if p.blocks]
    if meaningful_pages:
        doc.overall_confidence = mean([p.page_confidence for p in meaningful_pages])
    elif doc.pages:
        doc.overall_confidence = 0.0
    else:
        doc.overall_confidence = 0.0
    doc.review_required_pages = [p.page_no for p in doc.pages if p.needs_review]
    return doc
