# OCR Pipeline MVP v2 프로젝트 분석

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | `ocr_pipeline_mvp_v2_ok` |
| **목적** | PDF/이미지 문서를 OCR 처리 후 구조화된 검수 가능 산출물(JSON/MD/HTML)로 변환 |
| **핵심 가치** | 정확도 중심 처리 + 검수 가능성 확보 |
| **라인 수** | Python 23개 파일, 총 **2,773 라인** |
| **Git 이력** | 2 commits (초기 커밋 단계) |

## 2. 아키텍처 구조

```
run_ocr.py (진입점)
  └─ ocr_pipeline/
       ├─ cli.py            (49줄)  - argparse CLI
       ├─ config.py         (65줄)  - dataclass 설정 계층
       ├─ pipeline.py       (129줄) - 핵심 오케스트레이션
       ├─ schemas.py        (100줄) - OCRBlock/OCRPage/OCRDocument/InputProfile
       ├─ profile.py        (140줄) - 문서 프로파일링 & 라우팅 힌트
       ├─ preprocessing.py  (94줄)  - 이미지 전처리 (denoise/deskew/table line)
       ├─ quality.py        (57줄)  - confidence 기반 품질 평가 & review 플래그
       ├─ html_renderer.py  (106줄) - 검수용 HTML 렌더링
       ├─ review_overlay.py (41줄)  - bbox 오버레이 이미지 생성
       ├─ compare.py        (598줄) - 원본 vs OCR 비교 & 비교 HTML 생성
       ├─ ensemble.py       (11줄)  - 다중 엔진 결과 선택 (confidence 비교)
       ├─ table_refiner.py  (244줄) - 표 영역 재크롭 후 재인식
       ├─ table_cropper.py  (68줄)  - PDF에서 표 bbox 기반 이미지 크롭
       ├─ table_rebuilder.py(52줄)  - markdown table → HTML 변환
       ├─ pdf_utils.py      (93줄)  - PyMuPDF 기반 PDF 유틸리티
       ├─ engines/
       │    ├─ base.py              (18줄)  - ABC 인터페이스
       │    ├─ paddle_vl_engine.py  (676줄) - ★ 주력 엔진 (문서 구조 복원)
       │    ├─ paddle_ocr_engine.py (139줄) - 기본 OCR 엔진 (fallback)
       │    └─ surya_engine.py      (24줄)  - 미구현 (NotImplementedError)
       └─ utils/
            ├─ io.py   (38줄) - JSON 저장/로드
            └─ text.py (20줄) - 공통 텍스트 유틸
```

## 3. 실행 흐름

```
CLI 인자 파싱 → PipelineConfig 구성
  → profile_input()      # 문서 분류 (text_heavy/mixed/table_heavy)
  → preprocess_image()   # 이미지 입력 시에만 (PDF는 스킵)
  → _select_engine()     # auto 라우팅: paddle_vl > paddle_ocr > surya
  → engine.process()     # OCR 엔진 실행
  → TableRefiner.refine_document()  # 표 재인식 (crop → 재OCR → score 비교 교체)
  → score_document()     # confidence 재계산 & review 플래그
  → 출력 저장 (JSON/MD/HTML/Comparison/Overlay)
```

### 3.1 엔진 자동 선택 로직 (`pipeline.py` `_select_engine`)

```
requested == "paddle_vl"  → paddle_vl (없으면 RuntimeError)
requested == "paddle_ocr"  → paddle_ocr (없으면 RuntimeError)
requested == "surya"       → surya (없으면 RuntimeError)

requested == "auto":
  profile.routing_hint == "prefer_paddle_vl"  → paddle_vl
  profile.routing_hint == "prefer_paddle_ocr"  → paddle_ocr
  paddle_vl.available() AND (table_heavy/mixed/pdf/image/unknown) → paddle_vl
  paddle_ocr.available()  → paddle_ocr
  surya.available()       → surya
  전부 불가               → RuntimeError
```

### 3.2 입력별 처리 차이

| 입력 유형 | 전처리 | 엔진 처리 방식 |
|-----------|--------|----------------|
| 이미지 (.png/.jpg/...) | denoise, deskew, table line 강화 | 이미지 직접 OCR |
| PDF (text 레이어 있음) | 스킵 | PDF 직접 처리 (PaddleVL) |
| PDF (스캔/이미지) | 스킵 | PDF 직접 처리 (엔진 내부 렌더링) |

> 참고: PDF 입력은 파이프라인이 미리 이미지화하지 않고, 엔진이 PDF를 직접 처리한다.
> PaddleOCRAdapter는 내부에서 PDF→페이지별 PNG 변환 후 OCR을 수행한다.

## 4. 데이터 모델 (`schemas.py`)

```
InputProfile
  ├─ source_file, file_type, pdf_type, page_count
  ├─ has_text_layer, layout_type, table_complexity, quality_tier
  ├─ comparison_mode, routing_hint
  └─ probable_table_pages, languages, extra

OCRDocument
  ├─ document_id, source_file, pdf_type, page_count
  ├─ overall_confidence, review_required_pages, engine_used
  └─ pages: list[OCRPage], metadata

OCRPage
  ├─ page_no, width, height, source_image
  ├─ page_confidence, needs_review, review_reason
  └─ blocks: list[OCRBlock], raw_paths, extra

OCRBlock
  ├─ block_id, block_type, page_no, bbox
  ├─ text, html, markdown, confidence
  ├─ review, review_reason, source_engine
  └─ extra
```

block_type 종류: `text`, `title`, `table`, `formula`, `figure`

## 5. 설정 구조 (`config.py`)

```
PipelineConfig
  ├─ engine: "auto" | "paddle_vl" | "paddle_ocr" | "surya"
  ├─ language, device, workers
  ├─ merge_cross_page_tables, relevel_titles, concatenate_pages
  ├─ preprocess: PreprocessConfig
  │    ├─ enabled, deskew, denoise
  │    ├─ adaptive_threshold, strengthen_table_lines
  │    └─ line_kernel_scale
  ├─ review: ReviewConfig
  │    ├─ page_score_threshold (0.70)
  │    ├─ block_score_threshold (0.65)
  │    ├─ numeric_mismatch_penalty (0.10)
  │    └─ empty_table_penalty (0.20)
  ├─ output: OutputConfig
  │    ├─ save_raw, save_markdown, save_html
  │    ├─ save_comparison, save_overlay
  │    └─ save_intermediate_images
  └─ table_refine: TableRefineConfig
       ├─ enabled, engine, dpi, padding
       ├─ max_tables_per_page, min_crop_width, min_crop_height
       └─ prefer_larger_table
```

## 6. 엔진 상태

| 엔진 | 상태 | 역할 | 파일 |
|------|------|------|------|
| **PaddleVLAdapter** | ✅ 주력 | PDF 직접 처리, 표 병합, 제목 레벨링, HTML/Markdown fallback | `engines/paddle_vl_engine.py` (676줄) |
| **PaddleOCRAdapter** | ✅ Fallback | PDF→이미지 변환 후 기본 OCR (표 구조 약함) | `engines/paddle_ocr_engine.py` (139줄) |
| **SuryaAdapter** | ❌ 미구현 | `process()` 호출 시 `NotImplementedError` 발생 | `engines/surya_engine.py` (24줄) |

### 6.1 PaddleVLAdapter 상세 (프로젝트 핵심)

이 어댑터가 프로젝트의 실제 차별점이다. 단순 OCR이 아니라 문서 구조 복원까지 노린 구현이 여기에 모여 있다.

주요 기능:
- `PaddleOCRVL.predict()` + `restructure_pages()` 호출
- 표 병합 (`merge_tables`), 제목 레벨링 (`relevel_titles`), 페이지 연결 (`concatenate_pages`)
- Raw JSON 재귀 순회로 block 후보 추출 (`_extract_blocks`)
- `_extract_blocks` 내부 휴리스틱:
  - `block_type` / `type` / `label` / `category` 키에서 블록 유형 판별
  - `text` / `content` / `rec_text` / `markdown` 등에서 텍스트 추출
  - HTML 테이블, markdown 테이블, text 내 HTML fragment 감지로 표 블록 식별
- Markdown fallback → block 병합 로직 (`_merge_or_replace_with_markdown`)
  - JSON block이 부실할 때 page markdown에서 표/제목/텍스트를 보충
- **병합된 표 자동 분리** (`_split_merged_table_blocks`)
  - 반복되는 헤더 행(구분/계획/실적/금액 등)이 나타나는 지점에서 표 분할
- **PDF 텍스트 레이어 기반 표 좌상단 셀 보정** (`_repair_table_blocks_from_pdf_text`)
  - rowspan/colspan이 있는 큰 첫 셀이 OCR에서 잘린 경우, PDF 텍스트 단어 위치를 참고해 복원
- Deduplication: 동일 block_type + text + markdown + bbox 조합 제거

### 6.2 PaddleOCRAdapter 상세

- PDF 입력 시 `render_pdf_to_images()`로 페이지별 PNG 변환 후 OCR
- 결과는 주로 텍스트 블록 단위 (`rec_texts`, `rec_scores`, `dt_polys`)
- 표 구조 복원은 거의 하지 않음
- 안정적인 fallback 엔진 역할

### 6.3 SuryaAdapter

- `available()`만 구현, `process()`는 `NotImplementedError` 발생
- CLI에서 `--engine surya` 선택 가능하나 실행 시 실패

## 7. 핵심 기능 상세

### 7.1 문서 프로파일링 (`profile.py`)

```
profile_input(input_path)
  ├─ PDF인 경우:
  │    ├─ detect_pdf_text_layer() → has_text, pdf_type (text/image)
  │    ├─ get_pdf_page_count()
  │    ├─ extract_pdf_page_texts() → 페이지별 텍스트
  │    ├─ _classify_layout() → text_heavy / mixed / table_heavy
  │    │    ├─ 페이지별 테이블 점수 계산 (헤더 히트, 숫자 밀도, 짧은 줄 비율)
  │    │    ├─ table_page_ratio ≥ 0.6 또는 avg_score ≥ 2.2 → table_heavy
  │    │    ├─ avg_score ≥ 1.2 또는 테이블 페이지 존재 → mixed
  │    │    └─ 그 외 → text_heavy
  │    ├─ _classify_table_complexity() → none / simple / complex
  │    └─ _routing_hint() → prefer_paddle_vl / prefer_paddle_ocr
  └─ 이미지인 경우:
       └─ routing_hint = "prefer_paddle_ocr", layout_type = "unknown"
```

> **한계**: 언어는 항상 `["ko", "en"]` 고정값. 실제 자동 감지 미구현.

### 7.2 이미지 전처리 (`preprocessing.py`)

이미지 입력에만 적용 (PDF는 스킵):

1. **Denoise** - `cv2.fastNlMeansDenoising`
2. **Deskew** - 기울기 추정 (`estimate_skew_angle`) 후 회전 보정
3. **Adaptive Threshold** - 적응형 이진화 (기본 비활성)
4. **Table Line 강화** - 수평/수직 모폴로지 연산으로 표 선 강화

### 7.3 품질 평가 (`quality.py`)

```
score_block()
  ├─ 빈 콘텐츠 → -0.30, "empty_content"
  ├─ 표인데 HTML 없음 → -empty_table_penalty, "table_without_html"
  ├─ 반복 숫자 패턴 (3개 이상 동일) → -penalty/2, "repetitive_numeric_pattern"
  ├─ confidence < block_score_threshold → review=True, "low_block_confidence"
  └─ confidence 클램핑 [0.0, 1.0]

score_page()
  ├─ page_confidence = mean(block confidences)
  ├─ confidence < page_score_threshold → needs_review=True
  ├─ 블록 중 review가 하나라도 있으면 needs_review=True
  └─ 블록 없으면 "page_without_blocks"

score_document()
  ├─ 모든 페이지/블록에 대해 score_block → score_page 순서 적용
  ├─ overall_confidence = mean(page confidences)
  └─ review_required_pages 수집
```

### 7.4 표 재인식 (`table_refiner.py`)

```
TableRefiner.refine_document()
  ├─ PDF가 아니면 스킵
  ├─ 엔진 구성 (paddle_vl 또는 paddle_ocr)
  ├─ PDFTableCropper로 고해상도(400dpi) 페이지 렌더링
  └─ 페이지별 표 블록에 대해:
       ├─ 표 bbox 추정 (block bbox 또는 인접 제목 기반)
       ├─ crop_table()로 표 영역 크롭
       ├─ _run_table_engine()로 재인식
       ├─ 기존 표 vs 재인식 결과 score 비교
       └─ 새 결과가 더 나으면 교체
```

### 7.5 비교 시스템 (`compare.py`)

```
build_document_comparison()
  ├─ 텍스트 레이어 PDF:
  │    ├─ 원본 텍스트 vs OCR 텍스트를 SequenceMatcher로 비교
  │    ├─ 블록 단위 매칭 (window 기반 best match)
  │    ├─ similarity ≥ 0.97 → match, ≥ 0.85 → partial, 미만 → mismatch
  │    └─ OCR에 없는 원본 라인 → "missing_in_ocr"
  └─ 이미지/스캔 PDF:
       ├─ 텍스트 비교 불가 → review 플래그 기반
       └─ confidence < 0.75 또는 review=True인 블록을 "needs_review"로 표시

render_comparison_html()
  ├─ 원본 페이지 이미지 + OCR 결과 페이지를 나란히 표시
  ├─ 불일치 블록은 빨간 배경으로 하이라이트
  ├─ 텍스트 diff (del=빨강, ins=초록)
  └─ 불일치 블록 상세 목록
```

## 8. 출력 구조

```
output_dir/
  ├─ input_profile.json          # 문서 분류 결과
  ├─ result.pretty.json          # 최종 구조화 결과
  ├─ result.md                   # RAG/검색용 markdown
  ├─ result.html                 # 브라우저 검토용 HTML
  ├─ comparison/
  │    ├─ result.compare.json    # 비교 데이터
  │    ├─ result.compare.html    # 비교 검토 HTML
  │    └─ source_pages/          # 원본 PDF 렌더링 이미지
  ├─ review_overlays/
  │    └─ page_XXXX.png          # bbox 오버레이 이미지
  ├─ table_refine/
  │    ├─ table_refine_report.json
  │    ├─ rendered_pages/        # 고해상도 페이지 이미지
  │    ├─ crops/                 # 표 크롭 이미지
  │    └─ page_*/                # 재인식 결과
  └─ paddle_vl/ 또는 paddle_ocr/
       ├─ raw_paddle_vl/         # 엔진 원본 산출물
       │    └─ page_XXXX/
       │         ├─ *.json
       │         └─ *.md
       └─ document_profile.json
```

## 9. 의존성

### 필수
| 패키지 | 용도 |
|--------|------|
| `pymupdf>=1.24` | PDF 분석, 렌더링, 텍스트 추출 |
| `pillow>=10.3` | 이미지 크기 판별, 표 크롭 |
| `numpy>=1.26` | 이미지 전처리 연산 |
| `opencv-python-headless>=4.10` | 전처리, overlay |

### OCR 엔진 (별도 설치 필요)
| 패키지 | 설치 순서 |
|--------|-----------|
| `paddlepaddle` 또는 `paddlepaddle-gpu` | **반드시 먼저 설치** |
| `paddleocr[all]>=3.2.0` | PaddleOCR 기본 엔진 |
| `paddleocr[doc-parser]>=3.2.0` | PaddleOCR-VL (문서 구조 복원) |

> **중요**: Paddle 계열은 `requirements.txt`만으로 바로 동작하지 않는다.
> `paddlepaddle`을 별도로 먼저 설치해야 한다.

## 10. 장점

1. **명확한 모듈 분리**: 파이프라인/엔진/품질/렌더링/비교가 잘 분리됨
2. **dataclass 기반 구조**: 유지보수성 우수, 직렬화 간단
3. **Raw 결과 보존**: 엔진 원본 산출물을 그대로 저장하여 디버깅 가능
4. **다중 fallback**: PaddleVL raw JSON → markdown fallback → PaddleOCR
5. **검수 친화적 설계**: confidence 기반 review 플래그 + overlay + 비교 HTML
6. **표 특화 기능**: 병합 표 자동 분리, 고해상도 재크롭 재인식, PDF 텍스트 기반 보정
7. **유연한 설정**: dataclass 기반 config로 전처리/리뷰/출력/표 재인식 개별 제어
8. **Auto 라우팅**: 문서 성격에 따라 최적 엔진 자동 선택

## 11. 리스크 및 약점

| 분류 | 항목 | 심각도 | 설명 |
|------|------|--------|------|
| 런타임 오류 | `surya` 엔진 선택 시 `NotImplementedError` | 🔴 높음 | CLI에서 선택 가능하지만 실행 불가 |
| 테스트 부재 | 테스트 코드 0줄 | 🔴 높음 | 회귀 방지 불가, 리팩토링 시 위험 |
| 스키마 취약성 | PaddleVL raw JSON 스키마 변경 | 🟡 중간 | `_extract_blocks()` 재귀 파서가 버전 변경에 취약 |
| 예외 처리 | `except Exception: pass` 패턴 | 🟡 중간 | overlay 생성 실패 등이 조용히 무시됨 |
| 언어 감지 | 고정값 `["ko", "en"]` | 🟡 중간 | 실제 언어 감지 미구현 |
| 품질 평가 | heuristic 기반 | 🟡 중간 | 실제 정확도와 불일치 가능성 |
| PDF 전처리 | 스캔 PDF 전처리 미적용 | 🟡 중간 | 이미지 입력에만 전처리 적용 |
| 엔진 가용성 | import 성공 여부만 확인 | 🟢 낮음 | 런타임 환경 문제 지연 발견 |
| 템플릿 엔진 없음 | f-string 기반 HTML 생성 | 🟢 낮음 | compare.py만 598줄로 매우 큼 |
| 병렬 처리 | `workers: int = 1` (미사용) | 🟢 낮음 | config에 있지만 파이프라인에서 미사용 |

## 12. 기술 스택

| 카테고리 | 기술 |
|----------|------|
| 언어 | Python 3.8+ (type hints, `from __future__ import annotations`) |
| OCR 엔진 | PaddleOCR, PaddleOCR-VL (doc-parser) |
| PDF 처리 | PyMuPDF (fitz) |
| 이미지 처리 | OpenCV (headless), Pillow |
| CLI | argparse |
| 데이터 모델 | dataclass (`slots=True`) |
| HTML 생성 | Python f-string (템플릿 엔진 없음) |
| 직렬화 | json (EnhancedJSONEncoder) |
| 텍스트 비교 | difflib.SequenceMatcher |

## 13. 개선 우선순위 제안

### P0 (즉시 조치)
1. **SuryaAdapter 비활성화** - CLI choices에서 제거하거나 `available()`이 항상 `False` 반환하도록 수정
2. **`except Exception: pass` 제거** - 최소한 로깅 추가

### P1 (단기)
3. **최소 회귀 테스트 추가** - schema 직렬화, profile 분류, markdown→HTML 변환, quality scoring
4. **엔진별 raw output fixture** + block parsing 테스트 작성
5. **스캔 PDF 전처리 경로 추가** - 이미지 렌더링 후 전처리 → OCR 파이프라인

### P2 (중기)
6. **언어 자동 감지 도입** - `langdetect` 또는 OCR 엔진 내장 감지 활용
7. **텍스트 레이어 PDF 최적 경로** - OCR 없이 직접 텍스트 추출하는 옵션
8. **표 복원 고도화** - rowspan/colspan 후처리, 숫자 합계 검증
9. **예외 처리 체계화** - 커스텀 예외 계층 + 로깅 프레임워크

### P3 (장기)
10. **API 서버 또는 배치 파이프라인 인터페이스**
11. **병렬 처리 활성화** (현재 `workers` 미사용)
12. **Surya 엔진 실제 구현 또는 완전 제거**
13. **DB 적재 / 결과 관리 시스템**

## 14. 결론

이 프로젝트는 `문서 OCR 결과를 사람이 검수 가능한 구조화 출력으로 만든다`는 목적이 분명한 MVP다. 현재 완성도는 `실험용 프로토타입`보다 한 단계 높은 수준이며, 특히 **PaddleOCR-VL 결과를 구조화하고 fallback 처리하는 후처리 계층**이 핵심 가치다.

프로젝트의 중심은 OCR 자체보다 `엔진 결과를 정리하고 검수 가능한 산출물로 바꾸는 파이프라인`에 있다. `paddle_vl_engine.py`(676줄)와 `compare.py`(598줄) 두 파일이 전체 코드의 약 46%를 차지하며, 이 두 모듈이 프로젝트의 실질적 핵심이다.

Production 수준으로 전환하려면 테스트 추가, 미구현 엔진 정리, 엔진 출력 스키마 변화 대응 강화, 품질 평가 기준 검증이 반드시 필요하다.
