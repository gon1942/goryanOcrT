# OCR Pipeline MVP 분석

## 1. 프로젝트 개요

이 프로젝트는 PDF 또는 이미지 문서를 입력으로 받아 OCR을 수행하고, 결과를 JSON/Markdown/HTML로 저장하는 문서 OCR 파이프라인 MVP다. 핵심 목표는 단순 텍스트 추출보다 `정확도 중심 처리`와 `검수 가능성 확보`에 있다.

주요 특징:

- 엔진 자동 선택(`auto`)
- `PaddleOCR-VL` 우선, 불가 시 `PaddleOCR` fallback
- OCR 결과에 대한 confidence 기반 품질 점수 산정
- 리뷰가 필요한 페이지 및 블록 표시
- 검수용 오버레이 이미지 생성

## 2. 진입점과 실행 흐름

실행 진입점은 [run_ocr.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/run_ocr.py) 이며, 
실제 CLI는 [ocr_pipeline/cli.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/cli.py) 에 있다.

전체 흐름:

1. CLI 인자로 입력 파일, 출력 디렉토리, 엔진, 언어, 디바이스 등을 받는다.
2. `PipelineConfig`를 구성한다.
3. `OCRPipeline.run()`이 호출된다.
4. 입력 파일을 프로파일링한다.
5. 이미지 입력일 경우 전처리를 수행한다.
6. 엔진을 선택한다.
7. 선택된 엔진이 문서를 페이지 단위로 OCR 처리한다.
8. 문서/페이지/블록 confidence를 재계산하고 review 여부를 부여한다.
9. `result.pretty.json`, `result.md`, `result.html`, `review_overlays/`를 저장한다.

핵심 오케스트레이션은 [ocr_pipeline/pipeline.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/pipeline.py) 에 집중되어 있다.

## 3. 핵심 모듈 역할

### `ocr_pipeline/cli.py`

- argparse 기반 CLI
- 옵션:
  - `--engine`: `auto`, `paddle_vl`, `paddle_ocr`, `surya`
  - `--disable-preprocess`
  - `--no-merge-cross-page-tables`
  - `--no-relevel-titles`
  - `--concatenate-pages`

### `ocr_pipeline/config.py`

- dataclass 기반 설정 집합
- 전처리, 리뷰 기준, 출력 옵션을 별도 config로 분리
- MVP 수준에서 확장하기 쉬운 구조다

### `ocr_pipeline/profile.py`

- 입력 파일 유형을 판별
- PDF는 텍스트 레이어 존재 여부를 샘플 페이지 기준으로 판정
- 현재 언어는 고정적으로 `["ko", "en"]` 반환

### `ocr_pipeline/preprocessing.py`

- 이미지 입력에만 적용
- denoise, deskew, adaptive threshold, table line 강화 지원
- OpenCV 의존

### `ocr_pipeline/engines/*`

- OCR 엔진 어댑터 계층
- 공통 인터페이스는 [ocr_pipeline/engines/base.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/engines/base.py)

### `ocr_pipeline/quality.py`

- 블록 및 페이지 confidence 후처리
- 빈 콘텐츠, HTML 없는 표, 반복 숫자 패턴에 penalty 부여
- review 플래그를 문서 결과에 반영

### `ocr_pipeline/html_renderer.py`

- OCR 결과를 빠르게 검토 가능한 단일 HTML로 렌더링
- 표 HTML, markdown table fallback, 일반 텍스트를 함께 처리

### `ocr_pipeline/review_overlay.py`

- OCR block bbox를 원본 이미지에 덧그려 검수용 이미지 생성
- 리뷰 대상 블록은 빨간색, 일반 블록은 녹색 표시

## 4. 엔진별 구현 상태

### PaddleOCRAdapter

파일: [ocr_pipeline/engines/paddle_ocr_engine.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/engines/paddle_ocr_engine.py)

- 사용 가능 여부를 import 기반으로 판별
- PDF 입력이면 페이지별 PNG로 렌더링 후 OCR 수행
- 결과는 주로 텍스트 블록 단위로 정규화
- 표 구조 복원은 거의 하지 않음
- 안정적인 fallback 엔진 역할에 가깝다

### PaddleVLAdapter

파일: [ocr_pipeline/engines/paddle_vl_engine.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/engines/paddle_vl_engine.py)

- 현재 프로젝트의 주력 엔진
- PDF에 대해 `restructure_pages()`를 사용해 표 병합, 제목 레벨링, 페이지 연결 옵션을 적용
- raw JSON/Markdown 산출물을 저장
- raw object를 재귀 순회하며 block 후보를 추출
- table HTML 또는 markdown table을 우선 보존
- raw block이 부실할 경우 page markdown 기반 fallback 블록 생성

이 어댑터가 프로젝트의 실제 차별점이다. 단순 OCR이 아니라 문서 구조 복원까지 노린 구현이 여기에 모여 있다.

### SuryaAdapter

파일: [ocr_pipeline/engines/surya_engine.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/engines/surya_engine.py)

- `available()`만 있고 `process()`는 미구현
- README의 "추가 개발 필요" 항목과 일치
- 현재는 선택 가능 옵션처럼 보이지만 실제 실행은 불가능

## 5. 데이터 모델

파일: [ocr_pipeline/schemas.py](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/ocr_pipeline/schemas.py)

주요 모델:

- `InputProfile`: 입력 문서 특성
- `OCRBlock`: 텍스트/표/제목/수식/도형 블록
- `OCRPage`: 페이지 단위 결과
- `OCRDocument`: 전체 문서 결과

장점:

- 직렬화가 단순하다
- 엔진별 raw 메타데이터를 `extra`에 수용 가능하다
- 후속 저장소 적재나 API 응답으로 재사용하기 쉽다

## 6. 출력 구조

실행 결과는 출력 디렉토리에 다음 형태로 저장된다.

- `result.pretty.json`: 최종 구조화 결과
- `result.md`: 간단한 문서형 markdown
- `result.html`: 사람이 검토하기 쉬운 HTML
- `review_overlays/`: bbox 시각화 이미지
- `paddle_vl/` 또는 `paddle_ocr/`: 엔진 원본 산출물

이미 생성된 `output_input4/`, `output_input5/`는 샘플 실행 결과로 보이며, README의 설명과 실제 출력 구조가 대체로 일치한다.

## 7. 의존성 구조

파일: [requirements.txt](/home/gon/work/my/doc_ocr/ocr_pipeline_mvp/requirements.txt)

핵심 의존성:

- `pymupdf`: PDF 분석 및 렌더링
- `pillow`: 이미지 크기 판별
- `numpy`, `opencv-python-headless`: 전처리 및 overlay
- `paddleocr[all]`: OCR 엔진

주의점:

- Paddle 계열은 `requirements.txt`만으로 바로 동작하지 않는다.
- `paddlepaddle` 또는 `paddlepaddle-gpu`를 별도로 먼저 설치해야 한다.
- README가 이 제약을 명시하고 있어 문서와 구현은 일관적이다.

## 8. 아키텍처 평가

좋은 점:

- 파이프라인, 엔진, 품질 평가, 렌더링이 비교적 명확하게 분리되어 있다.
- dataclass 중심 구조라 유지보수가 쉽다.
- MVP임에도 raw 결과와 최종 결과를 모두 남겨 디버깅성이 좋다.
- `PaddleOCR-VL` 결과가 불안정할 때 markdown fallback을 두어 실용성을 확보했다.

제약 또는 약점:

- 테스트 코드가 없다.
- `SuryaAdapter`는 옵션만 있고 실제 동작하지 않는다.
- 예외 처리가 전반적으로 최소 수준이다. 예를 들어 overlay 생성 실패는 조용히 무시된다.
- `profile_input()`의 언어 추정은 실제 감지가 아니라 고정값이다.
- 품질 평가 로직은 heuristic 중심이라 실제 정확도와 완전히 일치하지 않을 수 있다.
- HTML 렌더러의 스타일은 기능 위주이며 시각적 완성도는 낮다.
- PDF 텍스트 레이어가 있어도 별도의 텍스트 추출 최적 경로로 가지 않고 OCR 중심 흐름을 유지한다.

## 9. 코드 기준의 실질적 리스크

1. `surya` 엔진 선택 시 런타임 예외가 발생한다.
2. `PaddleOCR-VL` raw JSON 스키마가 버전별로 바뀌면 `_extract_blocks()` 재귀 파서가 쉽게 흔들릴 수 있다.
3. markdown table 정규식 기반 복원은 복잡한 표에서 한계가 있다.
4. 이미지 전처리는 PDF에는 직접 적용되지 않는다. 스캔 PDF 품질 개선 여지가 남아 있다.
5. `available()`가 단순 import 성공 여부만 보기 때문에 런타임 환경 문제가 뒤늦게 드러날 수 있다.

## 10. 개선 우선순위 제안

우선순위가 높음:

1. `SuryaAdapter`를 비활성화하거나 실제 구현한다.
2. 최소한의 회귀 테스트를 추가한다.
3. 엔진별 샘플 raw output fixture를 두고 block parsing 테스트를 만든다.
4. PDF 이미지형 문서에 대한 전처리 전략을 보강한다.

다음 단계:

1. text-layer PDF는 OCR을 건너뛰고 직접 추출하는 최적 경로 추가
2. 표 복원 고도화 (`rowspan`, `colspan`, 숫자 검증)
3. 문서 언어 자동 감지
4. API 서버 또는 배치 파이프라인 인터페이스 추가

## 11. 결론

이 프로젝트는 `문서 OCR 결과를 사람이 검수 가능한 구조화 출력으로 만든다`는 목적이 분명한 MVP다. 현재 완성도는 `실험용 프로토타입`보다 한 단계 높은 수준이며, 특히 `PaddleOCR-VL` 결과를 구조화하고 fallback 처리하는 부분이 핵심 가치다.

반면 production 수준으로 보려면 다음이 반드시 필요하다:

- 테스트 추가
- 미구현 엔진 정리
- 엔진 출력 스키마 변화 대응 강화
- 품질 평가 기준 검증

분석 결과만 보면, 현재 프로젝트의 중심은 OCR 자체보다 `엔진 결과를 정리하고 검수 가능한 산출물로 바꾸는 후처리 계층`에 있다.
