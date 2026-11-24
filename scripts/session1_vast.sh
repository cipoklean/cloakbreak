#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CloakBreak — Vast.ai GPU Session 1 (Resumable, tmux-safe, token-safe)
# =============================================================================
# Recommended Vast.ai Docker image:
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   (ships Python 3.11 + torch==2.6.0 + torchaudio==2.6.0 matching chatterbox-tts==0.1.7)
# Recommended hardware:
#   1x RTX 3090 or RTX 4090 (24 GB VRAM), On-Demand, >= 6 vCPUs, >= 50 GB disk, >= 1000 Mbps down
#
# Modes:
#   SMOKE=1 bash scripts/session1_vast.sh   -> deps + preflight + <4 min end-to-end smoke only
#   bash scripts/session1_vast.sh           -> runs smoke automatically as Step 2b, then full Session 1
#   ALLOW_SHORTCUT_WARNING=1 ...            -> allow Step 4 to continue if shortcut_warning is true
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
echo "[session1] Dataset repo: ${HF_DATASET_REPO} | SMOKE=${SMOKE:-0}"
echo "[session1] Host vCPUs: $(nproc)"
echo "================================================================="

# Step 1: System + pinned lockfile dependencies + strict pip check
if [ ! -f logs/.s1_deps_done ]; then
  echo "[step 1/6] Installing apt packages (ffmpeg, espeak-ng, libsndfile1, tmux)..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng libsndfile1 tmux

  echo "[step 1/6] Installing full transitive lock (requirements-gpu.lock) with --no-deps..."
  python3 -m pip install --no-cache-dir --no-deps -r requirements-gpu.lock
  echo "[step 1/6] Verifying dependency consistency via pip check..."
  python3 -m pip check
  python3 -m spacy download en_core_web_sm --quiet || true
  touch logs/.s1_deps_done
else
  echo "[step 1/6] Dependencies already installed (logs/.s1_deps_done) — skipping."
fi

# Step 2: 60-second GPU preflight + full import sweep (fail fast before spending GPU-hours)
if [ ! -f logs/.s1_preflight_done ]; then
  echo "[step 2/6] Running preflight import sweep + GPU synthesis check..."
  PYTHONPATH=. python3 -m scripts.preflight_vast
  touch logs/.s1_preflight_done
else
  echo "[step 2/6] Preflight already passed (logs/.s1_preflight_done) — skipping."
fi

# Step 2b: End-to-end smoke pipeline (< 4 min) before launching full generation
if [ ! -f logs/.s1_smoke_e2e_done ]; then
  echo "[step 2b/6] Running end-to-end GPU smoke pipeline (< 4 min)..."
  bash scripts/smoke_e2e.sh
  touch logs/.s1_smoke_e2e_done
else
  echo "[step 2b/6] End-to-end smoke already passed (logs/.s1_smoke_e2e_done) — skipping."
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[session1] SMOKE=1 complete. Stopping before full build."
  exit 0
fi

# Step 3: Full dataset download + unwatermarked Chatterbox-Turbo + disjoint Kokoro-82M
# (generate_spoofs.py is clip-level resumable and aborts if > 5% clips fail)
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

# Step 4: Full shortcut audit (exits non-zero on shortcut_warning unless ALLOW_SHORTCUT_WARNING=1)
if [ ! -f logs/.s1_audit_done ]; then
  echo "[step 4/6] Running acoustic shortcut audit (dur/RMS/SNR/rolloff; fails on shortcut_warning)..."
  python3 -m src.eval \
    --shortcut-only \
    --fail-on-shortcut \
    --manifest data/cloakbreak_v1/manifest.jsonl \
    --max-per-slice 1000
  touch logs/.s1_audit_done
else
  echo "[step 4/6] Shortcut audit already completed (logs/.s1_audit_done) — skipping."
fi

# Step 5: 1-epoch Stage-1 smoke test on WavLM-Base-Plus (with parallel Opus cache + multi-worker DataLoader)
if [ ! -f logs/.s1_smoke_done ]; then
  echo "[step 5/6] Running 1-epoch WavLM-Base-Plus smoke test..."
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
