#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CloakBreak — Vast.ai GPU Session 1 (Resumable, tmux-safe, token-safe)
# =============================================================================
# Recommended Vast.ai Docker image:
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   (ships Python 3.11 + torch==2.6.0 + torchaudio==2.6.0 matching chatterbox-tts==0.1.7)
# Recommended hardware:
#   1x RTX 3090 (24 GB VRAM), On-Demand <= $0.25/hr, >= 50 GB disk, >= 1000 Mbps down
# Target budget: ~$0.50-$0.65 (~2.0-2.5 hrs). DESTROY instance immediately when done.
#
# Run INSIDE tmux so SSH drops on mobile data never kill progress:
#   tmux new -s cb
#   bash scripts/session1_vast.sh
# Reattach after disconnect:
#   tmux attach -t cb
# =============================================================================

# Never enable set -x globally so secrets cannot leak to logs.
set +x

if [ -z "${TMUX:-}" ] && [ "${ALLOW_NO_TMUX:-0}" != "1" ]; then
  echo "ERROR: Run this script inside tmux so mobile SSH drops cannot kill paid GPU work."
  echo "  1. apt-get update -qq && apt-get install -y -qq tmux  (if tmux is missing)"
  echo "  2. tmux new -s cb"
  echo "  3. bash scripts/session1_vast.sh"
  echo "(Override only if needed with ALLOW_NO_TMUX=1)"
  exit 1
fi

mkdir -p logs
LOG_FILE="logs/session1.log"

# Prompt securely for HF_TOKEN without echo or shell history exposure
if [ -z "${HF_TOKEN:-}" ]; then
  read -r -s -p "HF write token (input hidden): " HF_TOKEN
  echo ""
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN cannot be empty."
  exit 1
fi

HF_DATASET_REPO="${HF_DATASET_REPO:-cipoklean/cloakbreak-data-v1}"
export HF_DATASET_REPO

# Redirect all subsequent output to both screen and logs/session1.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================="
echo "[session1] Started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[session1] Dataset repo: ${HF_DATASET_REPO}"
echo "================================================================="

# Step 1: System + Python dependencies (resumable via marker)
if [ ! -f logs/.s1_deps_done ]; then
  echo "[step 1/5] Installing apt packages (ffmpeg, espeak-ng, libsndfile1, tmux)..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng libsndfile1 tmux

  echo "[step 1/5] Installing pinned GPU requirements (torch==2.6.0 already in image)..."
  python3 -m pip install --no-cache-dir -r requirements-gpu.txt
  python3 -m spacy download en_core_web_sm --quiet || true
  touch logs/.s1_deps_done
else
  echo "[step 1/5] Dependencies already installed (logs/.s1_deps_done) — skipping."
fi

# Step 2: 60-second GPU preflight (fail fast before spending GPU-hours)
if [ ! -f logs/.s1_preflight_done ]; then
  echo "[step 2/5] Running 60-second preflight check..."
  python3 -m scripts.preflight_vast
  touch logs/.s1_preflight_done
else
  echo "[step 2/5] Preflight already passed (logs/.s1_preflight_done) — skipping."
fi

# Step 3: Dataset download + Chatterbox-Turbo clones + held-out Kokoro-82M test set
# (generate_spoofs.py is also clip-level resumable if interrupted mid-step)
if [ ! -f logs/.s1_dataset_done ]; then
  echo "[step 3/5] Building CloakBreak v1 dataset (resumable per clip)..."
  python3 -m src.generate_spoofs \
    --source hf \
    --hf-dataset SpeechAntiSpoofingBenchmarks/InTheWild \
    --out-dir data/cloakbreak_v1 \
    --max-real 12000 \
    --max-chatterbox 3000 \
    --max-kokoro 600 \
    --use-turbo
  touch logs/.s1_dataset_done
else
  echo "[step 3/5] Dataset already built (logs/.s1_dataset_done) — skipping."
fi

# Step 4: 1-epoch Stage-1 smoke test on WavLM-Base-Plus
if [ ! -f logs/.s1_smoke_done ]; then
  echo "[step 4/5] Running 1-epoch WavLM-Base-Plus smoke test..."
  python3 -m src.train \
    --manifest data/cloakbreak_v1/manifest.jsonl \
    --out-dir checkpoints/smoke \
    --backbone-id microsoft/wavlm-base-plus \
    --epochs-stage1 1 \
    --epochs-stage2 0 \
    --batch-size 24 \
    --device cuda
  touch logs/.s1_smoke_done
else
  echo "[step 4/5] Smoke test already completed (logs/.s1_smoke_done) — skipping."
fi

# Step 5: Upload generated dataset & manifest to private HF dataset repo
if [ ! -f logs/.s1_upload_done ]; then
  echo "[step 5/5] Uploading dataset + manifest to private HF repo ${HF_DATASET_REPO}..."
  python3 - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_DATASET_REPO"]
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path="data/cloakbreak_v1",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload CloakBreak v1 dataset + manifest",
)
print(f"Upload complete: https://huggingface.co/datasets/{repo_id}", flush=True)
PY
  touch logs/.s1_upload_done
else
  echo "[step 5/5] HF upload already completed (logs/.s1_upload_done) — skipping."
fi

echo "================================================================="
echo "[session1] ALL STEPS COMPLETE at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[session1] Next: record Instance ID, active hours & cost in SUBMISSION_LOG.md,"
echo "[session1]       then DESTROY this Vast instance immediately."
echo "================================================================="
