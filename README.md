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
  --input ./samp/input7.pdf \
  --output-dir ./output_input7 \
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
- **PaddleOCR-VL LoRA 파인튜닝** (한국어 도메인 특화)
- **SLANet 테이블 구조 인식 미세조정**

추가 개발이 필요한 것:
- Surya 비교 판독 결합
- 셀 단위 grid 복원 강화
- rowspan/colspan 후처리 고도화
- 숫자/합계 검증기 강화
- DB 적재 / API 서버화

---

## 엔진 커스터마이징

PaddleOCR-VL(메인 OCR 엔진)과 SLANet(테이블 구조 인식)을 파인튜닝하여
한국어 비즈니스 문서 특화 성능을 향상시킬 수 있습니다.

### 아키텍처 개요

```
┌─────────────────────────────────────────────┐
│  PaddleOCR-VL-1.5-0.9B (958M params)       │
│  ┌─────────────────────────────────────┐    │
│  │  LoRA Adapter (4.5M params, 0.47%)  │    │
│  │  타겟: q/k/v/o_proj, gate/up/down  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Table Refiner: SLANet_plus                 │
│  테이블 셀 바운딩박스 인식 → 구조 복원       │
│  (PaddleX 공식 파인튜닝 지원)                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Post-processing (Python)                   │
│  rowspan/colspan 복원, 빈 셀 정리           │
│  PDF text layer 교차 검증                   │
└─────────────────────────────────────────────┘
```

### 요구사양

| 항목 | 최소 권장 |
|------|----------|
| GPU | NVIDIA RTX 3090 (24GB) 1장 |
| RAM | 16GB |
| 디스크 | 모델 캐시 ~4GB + 학습 데이터 ~1GB |
| Python | 3.10 ~ 3.12 |
| CUDA | 12.x |

### 파인튜닝 전용 패키지 설치

기존 PaddlePaddle/PaddleOCR 설치에 추가로 다음을 설치합니다.

```bash
pip install peft bitsandbytes transformers torch accelerate
```

> `peft`는 LoRA 어댑터 적용/학습에 사용합니다.
> `bitsandbytes`는 QLoRA(4-bit 양자화) 학습에 필요합니다 (VRAM 절약용).
> `transformers`/`torch`는 PaddleOCR-VL 모델이 내부적으로 PyTorch 기반이므로 필요합니다.

### Step 1: 학습 데이터 생성

기존 파이프라인 출력 결과에서 테이블 이미지와 HTML 정답 쌍을 추출합니다.
**파이프라인을 새로 실행하지 않고** 이미 생성된 `result.pretty.json`과
`table_refine/crops/`를 재사용합니다.

#### VL (PaddleOCR-VL) 학습 데이터

```bash
# 단일 출력 디렉토리에서 생성
python scripts/prepare_vl_training_data_lite.py \
    --input-dir output_input7_v8/ \
    --output-dir data/vl_training/

# 여러 출력 디렉토리를 한 번에 (glob 매칭)
python scripts/prepare_vl_training_data_lite.py \
    --input-dir output_input7_v8/ \
    --output-dir data/vl_training/ \
    --val-ratio 0.2
```

출력:
```
data/vl_training/
├── train.jsonl    # 학습용 (이미지 경로, 프롬프트, HTML 응답)
├── val.jsonl      # 검증용
```

JSONL 각 줄의 형식:
```json
{
  "image": "/절대경로/crop.png",
  "prompt": "Table Recognition:",
  "response": "<table border=1><tr><td>구분</td>...</tr></table>",
  "source": "input7.pdf",
  "page": 1,
  "block_id": "p1_b6"
}
```

#### SLANet 학습 데이터

```bash
python scripts/prepare_slanet_training_data_lite.py \
    --input-dir output_input7_v8/ \
    --output-dir data/slanet_training/
```

출력:
```
data/slanet_training/
├── train.txt      # PubTabTableRecDataset 포맷
├── val.txt
└── images/        # 테이블 crop 이미지 복사본
    ├── table_0000.png
    ├── table_0001.png
    └── ...
```

#### 전체 파이프라인 재실행으로 데이터 생성 (데이터 추가 시)

여러 PDF에서 한 번에 데이터를 모으려면:

```bash
# 테이블 전용
python scripts/prepare_vl_training_data.py \
    --input-dir samp/ \
    --output-dir data/vl_training_full/ \
    --table-only \
    --val-ratio 0.2

# 테이블 + 텍스트 모두
python scripts/prepare_vl_training_data.py \
    --input-dir samp/ \
    --output-dir data/vl_training_full/ \
    --val-ratio 0.2
```

> 이 스크립트는 내부에서 파이프라인을 실행하므로 시간이 오래 걸립니다.
> 이미 처리한 결과가 있다면 `*_lite.py`를 사용하세요.

### Step 2: PaddleOCR-VL LoRA 파인튜닝

```bash
# 기본 학습 (float16, gradient checkpointing)
python scripts/train_lora_vl.py \
    --data-dir data/vl_training/ \
    --output-dir models/lora_vl_v1/ \
    --epochs 10 \
    --batch-size 1 \
    --grad-accum 8 \
    --use-cp

# QLoRA (4-bit 양자화, VRAM 절약)
python scripts/train_lora_vl.py \
    --data-dir data/vl_training/ \
    --output-dir models/lora_vl_v1/ \
    --epochs 10 \
    --batch-size 1 \
    --grad-accum 8 \
    --use-qlora \
    --use-cp

# 이어서 학습 (checkpoint에서 재개)
python scripts/train_lora_vl.py \
    --data-dir data/vl_training/ \
    --output-dir models/lora_vl_v1/ \
    --resume-from models/lora_vl_v1/checkpoint-5/
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model-path` | `~/.paddlex/official_models/PaddleOCR-VL-1.5` | 베이스 모델 경로 |
| `--epochs` | 3 | 학습 에포크 수 |
| `--batch-size` | 1 | 배치 크기 (큰 이미지는 1 권장) |
| `--grad-accum` | 8 | 그래디언트 누적 스텝 (effective = batch x accum) |
| `--lr` | 2e-4 | 학습률 |
| `--lora-rank` | 8 | LoRA 랭크 |
| `--lora-alpha` | 16 | LoRA 스케일링 팩터 |
| `--max-length` | 4096 | 최대 시퀀스스 길이 |
| `--use-qlora` | false | 4-bit 양자화 사용 (VRAM ~12GB) |
| `--use-cp` | false | gradient checkpointing 사용 |

**VRAM 요구량:**
| 모드 | VRAM |
|------|------|
| float16 + grad checkpoint | ~16GB |
| QLoRA (4-bit) + grad checkpoint | ~12GB |
| float32 (테스트용) | ~20GB |

**출력 구조:**
```
models/lora_vl_v1/
├── final/                     # 최종 LoRA 어댑터
│   ├── adapter_config.json    # LoRA 설정
│   ├── adapter_model.safetensors  # 가중치 (~18MB)
│   └── tokenizer.model        # 토크나이저
├── checkpoint-1/               # 에포크별 체크포인트
├── checkpoint-2/
└── train_args.json            # 학습 인자 기록
```

### Step 3: SLANet 테이블 구조 인식 미세조정

```bash
python scripts/train_slanet.py \
    --data-dir data/slanet_training/ \
    --output-dir models/slanet_custom_v1/ \
    --model-name SLANet_plus \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4
```

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model-name` | SLANet_plus | 베이스 모델 (SLANet, SLANeXt_wired 등) |
| `--epochs` | 50 | 학습 에포크 수 |
| `--batch-size` | 4 | 배치 크기 |
| `--lr` | 1e-4 | 학습률 |
| `--max-len` | 1024 | 테이블 이미지 최대 리사이즈 길이 |
| `--gpu` | 0 | 사용할 GPU 디바이스 ID |

### Step 4: 파인튜닝된 모델로 추론

#### 방법 A: Config로 LoRA 활성화

```python
from ocr_pipeline.config import PipelineConfig

config = PipelineConfig()
config.lora.enabled = True
config.lora.adapter_path = "models/lora_vl_v1/final"
config.table_refine.model_path = "models/slanet_custom_v1/export"

# 파이프라인 실행
pipeline = OCRPipeline(config)
doc = pipeline.run("input.pdf", "output/")
```

#### 방법 B: Transformers로 LoRA 어댑터 직접 로드

PaddleOCR 파이프라인(PaddlePaddle 런타임)과는 별도로,
transformers 기반으로 LoRA 적용 모델을 직접 사용할 수 있습니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "~/.paddlex/official_models/PaddleOCR-VL-1.5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA 어댑터 결합
model = PeftModel.from_pretrained(model, "models/lora_vl_v1/final")

# 프로세서 로드
processor = AutoProcessor.from_pretrained(
    "~/.paddlex/official_models/PaddleOCR-VL-1.5",
    trust_remote_code=True,
)

# 추론
from PIL import Image
image = Image.open("table_crop.png")
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Table Recognition:"},
    ]},
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
result = processor.decode(output[0], skip_special_tokens=True)
```

### 학습 데이터 확보

현재 `samp/` 폴더의 10개 PDF에서 생성 가능한 학습 데이터량:

| PDF | 테이블 수 | 텍스트 블록 수 |
|-----|---------|---------------|
| input1 ~ input10 | ~30-60개 | ~50-100개 |

**권장:** 최소 100개 이상의 테이블 샘플로 학습하세요.
데이터가 부족하면 과적합이 발생합니다.

데이터를 늘리는 방법:
1. `samp/`의 다른 PDF로 `prepare_vl_training_data.py` 실행
2. 비슷한 도메인의 한국어 비즈니스 문서 추가
3. `--table-only` 없이 실행하면 텍스트 블록도 학습 데이터에 포함

### 스크립트 참고

```
scripts/
├── prepare_vl_training_data.py         # 파이프라인 실행 + 데이터 추출 (무거움)
├── prepare_vl_training_data_lite.py     # 기존 출력 재사용 (가벼움)
├── prepare_slanet_training_data.py      # 파이프라인 실행 + SLANet 데이터 (무거움)
├── prepare_slanet_training_data_lite.py # 기존 출력 재사용 (가벼움)
├── train_lora_vl.py                     # PaddleOCR-VL LoRA 학습
└── train_slanet.py                      # SLANet 미세조정
```

### 제한사항

1. **LoRA 추론 경로**: 현재 PaddleOCR-VL은 PaddlePaddle 런타임으로 실행됩니다.
   학습된 LoRA 어댑터를 PaddlePaddle 런타임에 직접 적용할 수 없습니다.
   LoRA 적용 추론은 transformers 기반 별도 경로를 사용해야 합니다
   (위 "방법 B" 참고).

2. **데이터 부족**: `samp/`의 PDF는 10개뿐이므로, LoRA 파인튜닝 효과가 제한적입니다.
   실제 도메인 문서 100+ 개에서 데이터를 확보하는 것을 권장합니다.

3. **PaddleX SLANet 학습**: PaddleX의 `build_trainer` API가 버전에 따라 달라질 수 있습니다.
   문제 발생 시 PaddleX CLI를 직접 사용하세요:
   ```bash
   paddlex train --model SLANet_plus --data data/slanet_training/ --output models/slanet_custom_v1/
   ```

---

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

### LoRA 학습 시 `CUDA out of memory`

해결:
- `--use-qlora` 추가 (4-bit 양자화로 VRAM 절약)
- `--use-cp` 추가 (gradient checkpointing)
- `--batch-size 1` 고정
- `--grad-accum`을 늘려 effective batch size 유지

### LoRA 학습 시 `transformers` 관련 오류

해결:
```bash
pip install --upgrade transformers peft bitsandbytes torch
```
