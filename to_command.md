# LoRA 학습 — 재부팅 후 실행 가이드

## 1. 재부팅

```bash
sudo reboot
```

## 2. 재부팅 후 학습 실행 (새 터미널)

```bash
cd /home/gon/work/my/doc_ocr/ocr_pipeline_mvp_v2_ok
CUDA_VISIBLE_DEVICES=1 python scripts/train_lora_vl.py \
  --data-dir data/vl_training/ \
  --output-dir models/lora_vl_v2b/ \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 2e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-length 4096 \
  --use-cp \
  --fp16 \
  --num-workers 2
```

예상 소요 시간: ~10~15분 (37샘플, 3 epoch)

## 3. 새 AI 세션에서 요청할 내용

학습이 완료된 후 새 opencode 세션을 열고 아래를 복사해서 붙여넣으세요:

```
LoRA 학습이 완료되었습니다. 결과를 검증해주세요.

- 학습 결과: models/lora_vl_v2b/final/
- 학습 데이터: data/vl_training/ (37샘플, 32 train + 5 val)
- 대상 모델: PaddleOCR-VL-1.5 (한국어 표 인식 LoRA 파인튜닝)
- train_lora_vl.py에 --fp16 옵션을 추가한 상태

확인할 것:
1. models/lora_vl_v2b/final/ 에 어댑터 파일이 정상 저장되었는지
2. loss 수렴 확인 (초기 loss vs 최종 loss vs eval loss)
3. 이전 결과 models/lora_vl_v1/final/ 과 비교
4. README.md 하단(560~577행)의 중복 흐름 설명 제거
```
