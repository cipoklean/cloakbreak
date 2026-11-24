#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CloakBreak — Vast.ai GPU Session 1 (Resumable, tmux-safe, token-safe)
# =============================================================================
# Recommended Vast.ai Docker image:
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   (ships Python 3.11 + torch==2.6.0 + torchaudio==2.6.0 matching chatterbox-tts==0.1.7)
# Recommended hardware:
#   1x RTX 3090 (24 GB VRAM), On-Demand <= $0.25/hr, >= 6 vCPUs, >= 50 GB disk, >= 1000 Mbps down
#
# Expected RTX 3090 throughput & Session 1 estimate (~0.7-1.2 hrs, ~$0.18-$0.30):
#   - Shuffled InTheWild (12k clips, seed=42):     ~3-6 min
#   - LibriSpeech clean stream (3.7k bonafide):    ~2-3 min (CC-BY-4.0, speaker-disjoint)
#   - Chatterbox-Turbo (3,000 unwatermarked clips): ~1.5-3.0 clips/sec (~17-35 min; warn < 0.5 clips/sec)
#   - Kokoro-82M (600 disjoint held-out clips):    ~10-20 clips/sec (~1 min)
#   - Shortcut audit (dur/RMS/SNR/rolloff check):  ~30 sec
#   - Opus 16k parallel CPU cache (6-12 vCPUs):    ~1.5-2.5 min (~150-300 clips/sec)
#   - WavLM Stage-1 1-epoch smoke test (bs=24):    ~4-8 steps/sec (~1.5-2.5 min)
#   - HF upload (generated clips + manifest only): ~30-60 sec (~350-500 MB)
#
# Run INSIDE tmux so SSH drops on mobile data never kill progress:
#   tmux new -s cb
#   bash scripts/session1_vast.sh
# Reattach after disconnect:
#   tmux attach -t cb
# =============================================================================

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
echo "[session1] Host vCPUs: $(nproc)"
echo "================================================================="

# Step 1: System + Python dependencies (resumable via marker)
if [ ! -f logs/.s1_deps_done ]; then
  echo "[step 1/6] Installing apt packages (ffmpeg, espeak-ng, libsndfile1, tmux)..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng libsndfile1 tmux

  echo "[step 1/6] Installing pinned GPU requirements (torch==2.6.0 already in image)..."
  python3 -m pip install --no-cache-dir -r requirements-gpu.txt
  python3 -m spacy download en_core_web_sm --quiet || true
  touch logs/.s1_deps_done
else
  echo "[step 1/6] Dependencies already installed (logs/.s1_deps_done) — skipping."
fi

# Step 2: 60-second GPU preflight (fail fast before spending GPU-hours)
if [ ! -f logs/.s1_preflight_done ]; then
  echo "[step 2/6] Running 60-second preflight check..."
  PYTHONPATH=. python3 -m scripts.preflight_vast
  touch logs/.s1_preflight_done
else
  echo "[step 2/6] Preflight already passed (logs/.s1_preflight_done) — skipping."
fi

# Step 3: Dataset download + unwatermarked Chatterbox-Turbo + disjoint Kokoro-82M
# (generate_spoofs.py is clip-level resumable if interrupted mid-step)
if [ ! -f logs/.s1_dataset_done ]; then
  echo "[step 3/6] Building CloakBreak v1 dataset (shuffled InTheWild + LibriSpeech clean + unwatermarked Chatterbox + disjoint Kokoro)..."
  python3 -m src.generate_spoofs \
    --source hf \
    --hf-dataset SpeechAntiSpoofingBenchmarks/InTheWild \
    --clean-real-dataset openslr/librispeech_asr \
    --text-dataset agentlans/high-quality-english-sentences \
    --out-dir data/cloakbreak_v1 \
    --max-real 12000 \
    --max-clean-train 2600 \
    --max-clean-val 500 \
    --max-clean-test 600 \
    --max-chatterbox 3000 \
    --max-kokoro 600 \
    --seed 42 \
    --use-turbo
  touch logs/.s1_dataset_done
else
  echo "[step 3/6] Dataset already built (logs/.s1_dataset_done) — skipping."
fi

# Step 4: Shortcut audit (disjoint sentences + trivial duration/RMS check)
if [ ! -f logs/.s1_audit_done ]; then
  echo "[step 4/6] Running duration + RMS shortcut audit & disjoint-sentence check..."
  python3 -m src.eval \
    --shortcut-only \
    --manifest data/cloakbreak_v1/manifest.jsonl \
    --max-per-slice 1000
  touch logs/.s1_audit_done
else
  echo "[step 4/6] Shortcut audit already completed (logs/.s1_audit_done) — skipping."
fi

# Step 5: 1-epoch Stage-1 smoke test on WavLM-Base-Plus (with parallel Opus cache + multi-worker DataLoader)
if [ ! -f logs/.s1_smoke_done ]; then
  echo "[step 5/6] Running 1-epoch WavLM-Base-Plus smoke test (expect ~4-8 steps/sec on RTX 3090)..."
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
  echo "[step 5/6] Smoke test already completed (logs/.s1_smoke_done) — skipping."
fi

# Step 6: Upload ONLY generated synthetic clips + prompts + manifest to private HF repo
# (InTheWild stays on HF upstream — avoids re-uploading ~3 GB and saves time)
if [ ! -f logs/.s1_upload_done ]; then
  echo "[step 6/6] Uploading generated spoofs + disjoint_prompts.json + manifest.jsonl to ${HF_DATASET_REPO}..."
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
    allow_patterns=["chatterbox/*", "kokoro_unseen/*", "disjoint_prompts.json", "manifest.jsonl"],
    commit_message="Upload CloakBreak v1 synthetic spoofs, disjoint prompts, and manifest",
)
print(f"Upload complete: https://huggingface.co/datasets/{repo_id}", flush=True)
PY
  touch logs/.s1_upload_done
else
  echo "[step 6/6] HF upload already completed (logs/.s1_upload_done) — skipping."
fi

echo "================================================================="
echo "[session1] ALL STEPS COMPLETE at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[session1] Next: record Instance ID, throughput, active hours & cost in SUBMISSION_LOG.md,"
echo "[session1]       then DESTROY this Vast instance immediately."
echo "================================================================="
