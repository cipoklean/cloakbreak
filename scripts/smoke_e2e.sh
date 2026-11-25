#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CloakBreak — End-to-End GPU Smoke Pipeline (< 4 min)
# Reusable across Sessions 1-4 before committing to multi-hour runs.
# Exercises: dataset build (40 InTheWild, 20 LibriSpeech, 6 Chatterbox, 4 Kokoro),
#            shortcut audit + augmented-window shortcut audit, 20 training steps
#            (Stage 1 + Stage 2), ONNX FP32 + INT8 quantization, ONNX eval pass,
#            and scratch HF upload.
# =============================================================================

set +x

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-data/smoke_v1}"
SMOKE_CKPT_DIR="${SMOKE_CKPT_DIR:-checkpoints/smoke_e2e}"
SMOKE_EVAL_JSON="${SMOKE_EVAL_JSON:-eval/smoke_results.json}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Cipoklean/cloakbreak-data-v1}"

echo "================================================================="
echo "[smoke] Starting end-to-end GPU smoke pipeline at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================="

echo "[smoke 1/5] Generating tiny dataset (40 InTheWild, 20 LibriSpeech, 6 Chatterbox, 4 Kokoro)..."
python3 -m src.generate_spoofs \
  --source hf \
  --hf-dataset SpeechAntiSpoofingBenchmarks/InTheWild \
  --clean-real-dataset openslr/librispeech_asr \
  --text-dataset agentlans/high-quality-english-sentences \
  --out-dir "$SMOKE_DATA_DIR" \
  --max-real 40 \
  --max-clean-train 12 \
  --max-clean-val 4 \
  --max-clean-test 4 \
  --max-chatterbox 6 \
  --max-kokoro 4 \
  --seed 42 \
  --use-turbo

echo "[smoke 2/5] Running shortcut audit on smoke manifest..."
ALLOW_SHORTCUT_WARNING=1 python3 -m src.eval \
  --shortcut-only \
  --manifest "$SMOKE_DATA_DIR/manifest.jsonl" \
  --max-per-slice 100

echo "[smoke 2b/5] Running augmented-window shortcut audit (pre-builds shared Opus cache)..."
ALLOW_SHORTCUT_WARNING=1 python3 -m src.eval \
  --shortcut-only \
  --augment-train-windows \
  --manifest "$SMOKE_DATA_DIR/manifest.jsonl" \
  --max-per-slice 100

echo "[smoke 3/5] Running 20-step WavLM train (Stage 1 + Stage 2) + ONNX FP32/INT8 export..."
python3 -m src.train \
  --manifest "$SMOKE_DATA_DIR/manifest.jsonl" \
  --out-dir "$SMOKE_CKPT_DIR" \
  --backbone-id microsoft/wavlm-base-plus \
  --epochs-stage1 1 \
  --epochs-stage2 2 \
  --max-steps 20 \
  --batch-size 4 \
  --device cuda

echo "[smoke 4/5] Running ONNX INT8 evaluation pass..."
python3 -m src.eval \
  --manifest "$SMOKE_DATA_DIR/manifest.jsonl" \
  --backend onnx \
  --model-path "$SMOKE_CKPT_DIR/cloakbreak_int8.onnx" \
  --max-per-slice 20 \
  --out-json "$SMOKE_EVAL_JSON"

if [ -n "${HF_TOKEN:-}" ] && [ "${SKIP_SMOKE_HF_UPLOAD:-0}" != "1" ]; then
  echo "[smoke 5/5] Testing HF Hub upload to scratch path smoke_check/ in ${HF_DATASET_REPO}..."
  python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_DATASET_REPO"]
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
smoke_eval = Path(os.environ.get("SMOKE_EVAL_JSON", "eval/smoke_results.json"))
smoke_manifest = Path(os.environ.get("SMOKE_DATA_DIR", "data/smoke_v1")) / "manifest.jsonl"
api.upload_file(
    path_or_fileobj=str(smoke_eval),
    path_in_repo="smoke_check/smoke_results.json",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Smoke pipeline check: eval results",
)
api.upload_file(
    path_or_fileobj=str(smoke_manifest),
    path_in_repo="smoke_check/manifest.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Smoke pipeline check: manifest",
)
print(f"[smoke] HF scratch upload OK: https://huggingface.co/datasets/{repo_id}", flush=True)
PY
else
  echo "[smoke 5/5] Skipping HF scratch upload (HF_TOKEN unset or SKIP_SMOKE_HF_UPLOAD=1)."
fi

echo "================================================================="
echo "[smoke] END-TO-END SMOKE PIPELINE PASSED at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================="
