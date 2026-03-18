# OCR Pipeline MVP

정확도와 검수 가능성을 우선한 문서 OCR 파이프라인입니다.

## 특징

- `auto` 라우팅: PDF/이미지 성격에 따라 엔진 선택
- 입력 문서를 `text_heavy` / `mixed` / `table_heavy` 로 분류하고 `input_profile.json` 저장
- `PaddleOCR-VL` 우선 사용, 없으면 `PaddleOCR`로 fallback
- 결과를 `result.pretty.json`, `result.md`, `result.html`로 동시 생성
- 가능한 경우 원본 문서와 OCR 결과를 비교한 `comparison/result.compare.html` 생성
- 검수용 `review_overlays/page_XXXX.png` 생성
- 텍스트 confidence + 표 복원 여부를 기준으로 `needs_review` 표기

## 폴더 구조

```text
ocr_pipeline_mvp/
├── ocr_pipeline/
│   ├── cli.py
│   ├── config.py
│   ├── engines/
│   ├── html_renderer.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── profile.py
│   ├── quality.py
│   ├── review_overlay.py
│   └── schemas.py
├── requirements.txt
└── run_ocr.py
```

## 설치 예시

> 중요: PaddleOCR 계열은 `paddleocr`만 설치해서는 부족할 수 있습니다. `paddlepaddle` 또는 `paddlepaddle-gpu`를 먼저 설치해야 합니다.

### 1) 가상환경

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2) PaddlePaddle 설치

공식 문서 기준으로 PaddleOCR-VL은 Python 3.8~3.12를 지원하며, 수동 설치 시 `PaddlePaddle`을 먼저 설치한 뒤 `paddleocr[doc-parser]`를 설치해야 합니다.

```bash
# CPU 예시
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 예시(CUDA 12.6 wheel)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

### 3) 프로젝트 의존성 설치

```bash
pip install -r requirements.txt
```

### 4) PaddleOCR-VL 포함 설치

```bash
pip install -U "paddleocr[doc-parser]"
```

### 5) 설치 확인

```bash
python -c "import paddle; print(paddle.__version__)"
```

## 실행 예시

### PDF 자동 라우팅

```bash
python run_ocr.py \
  --input ./input.pdf \
  --output-dir ./output_run1 \
  --engine auto \
  --device gpu
```

### PaddleOCR 기본 엔진 강제

```bash
python run_ocr.py \
  --input ./page1.png \
  --output-dir ./output_run2 \
  --engine paddle_ocr \
  --language korean \
  --device gpu
```

### PaddleOCR-VL 강제

```bash
export CUDA_VISIBLE_DEVICES=1
python run_ocr.py \
  --input ./samp/input1.pdf \
  --output-dir ./000_output \
  --engine paddle_vl \
  --device gpu

export CUDA_VISIBLE_DEVICES=1
python run_ocr.py \
  --input ./samp/input4.pdf \
  --output-dir ./output_input4 \
  --engine auto \
  --device gpu


PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python run_ocr.py \
  --input ./samp/input4.pdf \
  --output-dir ./output_input4 \
  --engine paddle_vl \
  --device gpu

```

--engine auto 는 입력 문서 성격과 설치된 엔진 상태를 보고 파이프라인이 엔진을 스스로 고릅니다.
반대로 --engine paddle_vl, --engine paddle_ocr, --engine surya 는 사용 엔진을 강제로 고정합니다.

현재 코드 기준 선택 규칙은 ocr_pipeline/pipeline.py 에 있습니다.

auto일 때 동작:

PaddleOCR-VL 이 설치되어 있고
입력이 PDF이거나, PDF 프로파일이 image 또는 unknown 이면
paddle_vl 선택
그게 안 되면 paddle_ocr
그것도 안 되면 surya
셋 다 안 되면 오류
즉, 지금 예시처럼 ./samp/input4.pdf 에 --engine auto 를 주면, 보통은 paddle_vl 이 먼저 선택됩니다.

차이를 실무적으로 보면:

--engine auto

환경에 따라 알아서 최선 엔진 선택
배치 실행에 편함
같은 명령이라도 설치 상태가 달라지면 실제 사용 엔진이 달라질 수 있음
--engine paddle_vl

항상 문서 구조 복원 중심 엔진 강제
표/제목/문서 구조가 중요한 PDF에 유리
설치 안 되어 있으면 바로 실패
--engine paddle_ocr

항상 기본 OCR 엔진 강제
단순 텍스트 추출 fallback 용도
표 구조나 문서 레이아웃 복원은 상대적으로 약함
--engine surya

현재 이 프로젝트에서는 실제 미구현
선택은 가능하지만 실행 시 실패
정리하면:

auto = 조건에 따라 엔진 자동 선택
그 외 값 = 해당 엔진을 무조건 사용
현재 프로젝트에서는 PDF 입력이면 auto 와 paddle_vl 결과가 같아질 가능성이 높습니다.



## 출력 파일

- `result.pretty.json`: 구조화된 최종 결과
- `input_profile.json`: 문서 분류 결과와 auto 라우팅 힌트
- `result.md`: RAG/검색용 markdown
- `result.html`: 브라우저로 확인 가능한 복원 결과
- `comparison/result.compare.json`: 원본 대비 비교 결과
- `comparison/result.compare.html`: 원본과 OCR 결과 차이 검토용 HTML
- `review_overlays/*.png`: 검토 필요 영역 표시 이미지
- `paddle_vl/` 또는 `paddle_ocr/`: 엔진 원본 산출물

## 현재 범위

이 버전은 **MVP** 입니다.

지원하는 것:
- 문서 프로파일링
- 기본 전처리
- PaddleOCR / PaddleOCR-VL 어댑터
- confidence 기반 검토 플래그
- HTML/MD/JSON 출력

추가 개발이 필요한 것:
- Surya 비교 판독 결합
- 셀 단위 grid 복원 강화
- rowspan/colspan 후처리 고도화
- 숫자/합계 검증기 강화
- DB 적재 / API 서버화

## 자주 발생하는 오류

### `ModuleNotFoundError: No module named 'paddle'`

원인: `paddleocr`는 설치되었지만 `paddlepaddle` 또는 `paddlepaddle-gpu`가 설치되지 않은 상태입니다.

해결 예시:

```bash
# CPU
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 또는 GPU(CUDA 12.6 예시)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

python -m pip install -U "paddleocr[doc-parser]"
python -c "import paddle; print(paddle.__version__)"
```

GPU wheel이 맞지 않으면 임시로 CPU 설치 후 `--device cpu`로 실행해도 됩니다.
