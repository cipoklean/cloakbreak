#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CloakBreak — Vast.ai GPU Session 2 (restore dataset -> gate -> train -> eval)
# =============================================================================
# Recommended Vast.ai Docker image:
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Recommended hardware:
#   1x RTX 3090 (24 GB VRAM), On-Demand, >= 6 vCPUs, >= 50 GB disk, >= 1000 Mbps down
# Budget: ~$2.82 total (MAX_HOURS=3.0 default + eval reserve; preemption-safe resume).
#
# Usage:
#   SMOKE=1 bash scripts/session2_vast.sh   -> few-minute end-to-end check first
#   bash scripts/session2_vast.sh           -> full Session 2 run
#   bash scripts/session2_vast.sh           -> rerun the SAME command after any
#                                               drop/preemption to resume
#                                               (--resume-from auto + HF ckpts)
#
# Env vars (all optional):
#   MAX_HOURS=3.0            total billed-hours budget for this box (session clock
#                            persists across reruns via $LOG_DIR/.s2_started_at)
#   EVAL_RESERVE_MIN=30      minutes reserved at the end for export/parity/eval/upload
#   MAX_STEPS / --max-steps N  cap optimizer steps (SMOKE defaults to 20)
#   DRY_RUN=1 / --dry-run    no HF token needed, cpu device, verify-only restore,
#                            --tiny-test model, dry-run artifact upload plan
#   SMOKE=1                  small ckpt/results dirs + smoke_check/ HF prefix
#   SMOKE_MAX_MIN=10         wall-clock cap (minutes) for SMOKE steps 3-7; the
#                            smoke clock starts AFTER shared setup steps 1-2
#                            (deps/preflight + dataset restore). On cap: exit 5.
#   RESUME=auto|none|path    checkpoint resume spec (default auto)
#   EXPECTED_MANIFEST_ROWS=N hard-check manifest row count after restore
#   VAST_RATE_PER_HR=X       prints projected training cost in [eta] lines
#   SKIP_DEPS=1              skip step-1 dependency install (DRY_RUN implies it)
#   HF_DATASET_REPO / HF_MODEL_REPO / BASELINE_ID / INT8_PARITY_MAX_DIFF
#                          dataset repo (default Cipoklean/cloakbreak-data-v1),
#                          model repo (default Cipoklean/cloakbreak-wavlm),
#                          baseline HF id (default Vansh180/deepfake-audio-wav2vec2,
#                          empty in DRY_RUN), INT8-vs-FP32 AUROC tolerance (0.01)
#
# Vast auto-starts a tmux session when you SSH in — run the script inside it so
# SSH drops on mobile data never kill progress:
#   bash scripts/session2_vast.sh
# After a reconnect:  tmux attach
# =============================================================================

set +x
cd "$(dirname "$0")/.."

# ---- CLI args ---------------------------------------------------------------
DRY_RUN_ARG=0
MAX_STEPS_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN_ARG=1; shift ;;
    --max-steps) MAX_STEPS_ARG="${2:?--max-steps needs a value}"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'"; exit 1 ;;
  esac
done

# ---- env / mode -------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
[ "$DRY_RUN_ARG" = "1" ] && DRY_RUN=1
SMOKE="${SMOKE:-0}"
MAX_HOURS="${MAX_HOURS:-3.0}"
EVAL_RESERVE_MIN="${EVAL_RESERVE_MIN:-30}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_DIR="${DATA_DIR:-data/cloakbreak_v1}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Cipoklean/cloakbreak-data-v1}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Cipoklean/cloakbreak-wavlm}"
BASELINE_ID="${BASELINE_ID:-Vansh180/deepfake-audio-wav2vec2}"
INT8_PARITY_MAX_DIFF="${INT8_PARITY_MAX_DIFF:-0.01}"
RESUME="${RESUME:-auto}"
EXPECTED_MANIFEST_ROWS="${EXPECTED_MANIFEST_ROWS:-}"
VAST_RATE_PER_HR="${VAST_RATE_PER_HR:-}"
SKIP_DEPS="${SKIP_DEPS:-0}"
SMOKE_MAX_MIN="${SMOKE_MAX_MIN:-10}"
SMOKE_CLOCK_START=0
SMOKE_DEADLINE=0
SMOKE_SETUP_MIN="0.0"
TRAIN_CMD_DEADLINE=""

if [ "$DRY_RUN" = "1" ]; then
  DEVICE="cpu"
  SKIP_DEPS=1
  BASELINE_ID=""
fi
if [ -z "${CKPT_DIR:-}" ]; then
  CKPT_DIR="checkpoints/run1"
  [ "$SMOKE" = "1" ] && CKPT_DIR="checkpoints/s2_smoke"
fi
if [ -z "${RESULTS_DIR:-}" ]; then
  RESULTS_DIR="results"
  [ "$SMOKE" = "1" ] && RESULTS_DIR="results/smoke"
fi
MP=".s2"; [ "$SMOKE" = "1" ] && MP=".s2smoke"
HF_PREFIX="checkpoints"; [ "$SMOKE" = "1" ] && HF_PREFIX="smoke_check/checkpoints"
UPLOAD_PREFIX=""; [ "$SMOKE" = "1" ] && UPLOAD_PREFIX="smoke_check/"
GATE_MAX_PER_SLICE=1000; [ "$SMOKE" = "1" ] && GATE_MAX_PER_SLICE=100
EVAL_MAX_PER_SLICE=600; [ "$SMOKE" = "1" ] && EVAL_MAX_PER_SLICE=20

if [ -z "${TMUX:-}" ] && [ "${ALLOW_NO_TMUX:-0}" != "1" ]; then
  echo "ERROR: Run this script inside the tmux session Vast opens on SSH login"
  echo "  (so mobile SSH drops cannot kill paid GPU work)."
  echo "  1. SSH into the instance (Vast auto-starts tmux for you)"
  echo "  2. bash scripts/session2_vast.sh"
  echo "  After a reconnect run: tmux attach"
  echo "(Override only if needed with ALLOW_NO_TMUX=1)"
  exit 1
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"
export HF_DATASET_REPO HF_MODEL_REPO
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Effective CPU count respects cgroup quotas (os.cpu_count()/nproc report the
# host's vCPUs on quota-limited Vast boxes and oversubscribe worker pools).
CB_CPUS="$("$PYTHON" -m src.sysinfo 2>/dev/null || nproc)"
CB_THREADS=$(( CB_CPUS < 8 ? CB_CPUS : 8 ))
export OMP_NUM_THREADS="$CB_THREADS" MKL_NUM_THREADS="$CB_THREADS" \
  OPENBLAS_NUM_THREADS="$CB_THREADS" NUMEXPR_NUM_THREADS="$CB_THREADS" \
  TOKENIZERS_PARALLELISM=false
echo "[session2] cgroup CPUs=$CB_CPUS threads=$CB_THREADS"

# Grep-able [KEY] summary lines: printed to the log AND appended to
# $RESULTS_DIR/key_lines.txt (uploaded with results/ by step 7 / budget stop).
# The file is append-only so reruns accumulate lines across invocations.
KEY_LINES_FILE="$RESULTS_DIR/key_lines.txt"
export CB_KEY_LINES_FILE="$KEY_LINES_FILE"

key_line() {
  local line="[KEY] $*"
  echo "$line"
  echo "$line" >> "$KEY_LINES_FILE"
}

smoke_min_fields() {
  # Prints "used_min=<u.u> cap_min=<c.c> setup_min=<s.s>" for the smoke clock.
  local now_s used cap
  now_s="$(date +%s)"
  used="$(awk "BEGIN{printf \"%.1f\", ($now_s - $SMOKE_CLOCK_START)/60.0}")"
  cap="$(awk "BEGIN{printf \"%.1f\", $SMOKE_MAX_MIN}")"
  echo "used_min=$used cap_min=$cap setup_min=$SMOKE_SETUP_MIN"
}

smoke_stop() {
  # $1 = step number where the cap hit. Exit 5: no upload, no SMOKE COMPLETE.
  key_line "smoke $(smoke_min_fields) complete=false"
  echo "[smoke] TIME CAP REACHED at step $1 — smoke did NOT complete; inspect the log before the full run"
  exit 5
}

run_capped() {
  # $1 = step number; remaining args = command. In SMOKE mode wraps the command
  # in `timeout --kill-after=30 <seconds until SMOKE_DEADLINE>`; a <=0 remaining
  # budget or a 124/137 timeout becomes a smoke cap stop (exit 5).
  local step_no="$1"; shift
  if [ "$SMOKE" != "1" ]; then
    "$@"
    return $?
  fi
  local remaining rc=0
  remaining=$(( SMOKE_DEADLINE - $(date +%s) ))
  if [ "$remaining" -le 0 ]; then
    smoke_stop "$step_no"
  fi
  timeout --kill-after=30 "$remaining" "$@" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    smoke_stop "$step_no"
  fi
  return "$rc"
}

emit_gate_key() {
  # [KEY] gate augmented_auroc=<a> eer=<e> pass=<bool> from newest audit JSON.
  local f
  f="$(ls -t "$RESULTS_DIR"/shortcut_audit_augmented_*.json 2>/dev/null | head -1 || true)"
  [ -z "$f" ] && return 0
  "$PYTHON" - "$f" <<'PY' || true
import json, os, sys

d = json.load(open(sys.argv[1]))
aug = d.get("augmented_train_window_shortcuts") or {}

def _fmt(v, nd):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if f != f:
        return "NA"
    return f"{f:.{nd}f}"

auroc, eer, warn = aug.get("auroc"), aug.get("eer_pct"), aug.get("shortcut_warning")
if auroc is None or eer is None:
    sys.exit(0)
line = (f"[KEY] gate augmented_auroc={_fmt(auroc, 4)} eer={_fmt(eer, 2)} "
        f"pass={'false' if warn else 'true'}")
print(line)
kf = os.environ.get("CB_KEY_LINES_FILE", "").strip()
if kf:
    with open(kf, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
PY
}

emit_int8_key() {
  # [KEY] int8_parity fp32_auroc=<a> int8_auroc=<b> delta=<d> from newest parity JSON.
  local f
  f="$(ls -t "$RESULTS_DIR"/int8_parity_*.json 2>/dev/null | head -1 || true)"
  [ -z "$f" ] && return 0
  "$PYTHON" - "$f" <<'PY' || true
import json, os, sys

d = json.load(open(sys.argv[1]))
f32, i8 = d.get("fp32") or {}, d.get("int8") or {}

def _fmt(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if f != f:
        return "NA"
    return f"{f:.{nd}f}"

if f32.get("auroc") is None or i8.get("auroc") is None:
    sys.exit(0)
line = (f"[KEY] int8_parity fp32_auroc={_fmt(f32['auroc'])} "
        f"int8_auroc={_fmt(i8['auroc'])} delta={_fmt(d.get('auroc_abs_diff'))}")
print(line)
kf = os.environ.get("CB_KEY_LINES_FILE", "").strip()
if kf:
    with open(kf, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
PY
}

emit_headline_key() {
  # [KEY] headline unseen_kokoro_opus16k eer=<e> auroc=<a> baseline_eer=<b|NA>
  # from $RESULTS_DIR/eval_results.json.
  local f="$RESULTS_DIR/eval_results.json"
  [ -f "$f" ] || return 0
  "$PYTHON" - "$f" <<'PY' || true
import json, os, sys

d = json.load(open(sys.argv[1]))
sl = d.get("unseen_kokoro_opus_16k") or {}
cb = sl.get("cloakbreak") or {}
bl = sl.get("baseline") or {}

def _fmt(v, nd):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if f != f:
        return "NA"
    return f"{f:.{nd}f}"

eer, auroc = cb.get("eer_pct"), cb.get("auroc")
if eer is None or auroc is None:
    sys.exit(0)
be = _fmt(bl.get("eer_pct"), 2) if bl.get("eer_pct") is not None else "NA"
line = (f"[KEY] headline unseen_kokoro_opus16k eer={_fmt(eer, 2)} "
        f"auroc={_fmt(auroc, 4)} baseline_eer={be}")
print(line)
kf = os.environ.get("CB_KEY_LINES_FILE", "").strip()
if kf:
    with open(kf, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
PY
}

# Session clock: epoch of FIRST launch on this box; reruns reuse it so the
# MAX_HOURS budget spans preemptions/restarts.
S2_STARTED_FILE="$LOG_DIR/.s2_started_at"
if [ -f "$S2_STARTED_FILE" ]; then
  SESSION_START_AT="$(cat "$S2_STARTED_FILE")"
else
  SESSION_START_AT="$(date +%s)"
  echo "$SESSION_START_AT" > "$S2_STARTED_FILE"
fi
SESSION_DEADLINE=$(( SESSION_START_AT + $(awk "BEGIN{print int($MAX_HOURS*3600)}") ))
TRAIN_DEADLINE=$(( SESSION_DEADLINE - $(awk "BEGIN{print int($EVAL_RESERVE_MIN*60)}") ))

INVOCATION_START="$(date +%s)"

# Prompt securely for HF_TOKEN without echo or shell history exposure
if [ "$DRY_RUN" != "1" ] && [ -z "${HF_TOKEN:-}" ]; then
  read -r -s -p "HF write token (input hidden): " HF_TOKEN
  echo ""
  export HF_TOKEN
fi

# Redirect all subsequent output to both screen and $LOG_DIR/session2.log
exec > >(tee -a "$LOG_DIR/session2.log") 2>&1

if [ "$DRY_RUN" != "1" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN cannot be empty."
    exit 1
  fi
  # Fail fast (stdlib-only) if the HF token/namespaces cannot upload.
  "$PYTHON" -m scripts.hf_auth_check
fi

echo "================================================================="
echo "[session2] Started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[session2] Dataset repo: ${HF_DATASET_REPO} | Model repo: ${HF_MODEL_REPO}"
echo "[session2] SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} MAX_HOURS=${MAX_HOURS} RESUME=${RESUME}"
echo "[session2] Session deadline (UTC): $(date -u -d "@$SESSION_DEADLINE" '+%Y-%m-%d %H:%M:%S') | train deadline (UTC): $(date -u -d "@$TRAIN_DEADLINE" '+%Y-%m-%d %H:%M:%S')"
echo "[session2] Host vCPUs: $(nproc) | cgroup-effective CPUs: $CB_CPUS | threads: $CB_THREADS"
echo "================================================================="

# ---- budget helpers ---------------------------------------------------------
upload_artifacts() {
  # Upload whatever exists (checkpoints/results/logs); never fails the run.
  "$PYTHON" -m src.hub upload-artifacts \
    --repo "$HF_MODEL_REPO" \
    --ckpt-dir "$CKPT_DIR" \
    --results-dir "$RESULTS_DIR" \
    --log-dir "$LOG_DIR" \
    --prefix "$UPLOAD_PREFIX" || echo "[warn] artifact upload failed (continuing)"
}

budget_stop() {
  if [ "$DRY_RUN" != "1" ]; then
    echo "[budget] uploading whatever artifacts exist before stopping..."
    upload_artifacts
  fi
  echo "[budget] STOPPED CLEANLY: MAX_HOURS=${MAX_HOURS} reached. Checkpoint is on HF; rerun to resume. DESTROY the instance if you are done."
  exit 0
}

check_budget() {
  if [ "$(date +%s)" -gt "$SESSION_DEADLINE" ]; then
    budget_stop
  fi
}

# ---- step 1: deps + preflight (shared markers) -------------------------------
if [ "$DRY_RUN" = "1" ] || [ "$SKIP_DEPS" = "1" ]; then
  echo "[step 1/8] SKIP_DEPS/DRY_RUN — skipping dependency install + preflight."
else
  LOCK_HASH="$(sha256sum requirements-gpu.lock | cut -d' ' -f1)"
  S2_MARKER_HASH="$(cat "$LOG_DIR/.s2_deps_done" 2>/dev/null || true)"
  if [ ! -f "$LOG_DIR/.s2_deps_done" ] || [ "$S2_MARKER_HASH" != "$LOCK_HASH" ]; then
    if [ -f "$LOG_DIR/.s2_deps_done" ]; then
      echo "[step 1/8] requirements-gpu.lock changed (marker ${S2_MARKER_HASH:0:12} -> ${LOCK_HASH:0:12}) — reinstalling dependencies."
    fi
    echo "[step 1/8] Installing apt packages (ffmpeg, espeak-ng, libsndfile1, tmux)..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng libsndfile1 tmux
    echo "[step 1/8] Installing full transitive lock (requirements-gpu.lock) with --no-deps..."
    "$PYTHON" -m pip install --no-cache-dir --no-deps -r requirements-gpu.lock
    echo "[step 1/8] Verifying dependency consistency via pip check..."
    "$PYTHON" -m pip check
    "$PYTHON" -m spacy download en_core_web_sm --quiet || true
    echo "$LOCK_HASH" > "$LOG_DIR/.s2_deps_done"
    rm -f "$LOG_DIR/.s2_preflight_done"
    echo "[step 1/8] Dependencies (re)installed; cleared .s2_preflight_done so preflight reruns."
  else
    echo "[step 1/8] Dependencies already installed and requirements-gpu.lock unchanged (${LOCK_HASH:0:12}) — skipping."
  fi

  if [ ! -f "$LOG_DIR/.s2_preflight_done" ]; then
    echo "[step 1/8] Running preflight import sweep + GPU check (--skip-tts)..."
    PYTHONPATH=. "$PYTHON" -m scripts.preflight_vast --skip-tts
    touch "$LOG_DIR/.s2_preflight_done"
  else
    echo "[step 1/8] Preflight already passed ($LOG_DIR/.s2_preflight_done) — skipping."
  fi
fi

# ---- step 2: restore dataset from HF repo + re-materialize real clips --------
if [ ! -f "$LOG_DIR/.s2_data_done" ]; then
  echo "[step 2/8] Restoring dataset -> $DATA_DIR (synthetic from HF snapshot; real clips re-materialized)..."
  restore_args=(--repo "$HF_DATASET_REPO" --out-dir "$DATA_DIR")
  if [ -n "$EXPECTED_MANIFEST_ROWS" ]; then
    restore_args+=(--expected-rows "$EXPECTED_MANIFEST_ROWS")
  fi
  if [ "$DRY_RUN" = "1" ]; then
    restore_args+=(--skip-download --skip-upstream)
  fi
  "$PYTHON" -m src.restore_dataset "${restore_args[@]}"
  touch "$LOG_DIR/.s2_data_done"
else
  echo "[step 2/8] Dataset already restored ($LOG_DIR/.s2_data_done) — skipping."
fi

# Smoke wall-clock: starts AFTER shared setup (steps 1-2) so deps/restore time —
# which the full run reuses via markers — does not count against the smoke cap.
TRAIN_CMD_DEADLINE="$TRAIN_DEADLINE"
if [ "$SMOKE" = "1" ]; then
  SMOKE_CLOCK_START="$(date +%s)"
  SMOKE_SETUP_MIN="$(awk "BEGIN{printf \"%.1f\", ($SMOKE_CLOCK_START - $INVOCATION_START)/60.0}")"
  SMOKE_DEADLINE=$(( SMOKE_CLOCK_START + $(awk "BEGIN{print int($SMOKE_MAX_MIN*60)}") ))
  if [ "$SMOKE_DEADLINE" -lt "$TRAIN_CMD_DEADLINE" ]; then
    TRAIN_CMD_DEADLINE="$SMOKE_DEADLINE"
  fi
  echo "[smoke] wall-clock cap ${SMOKE_MAX_MIN} min started at $(date -u -d "@$SMOKE_CLOCK_START" '+%Y-%m-%d %H:%M:%S UTC') (setup steps 1-2 took ${SMOKE_SETUP_MIN} min)"
fi

# ---- step 3: augmented-window shortcut gate (before any GPU training) --------
if [ ! -f "$LOG_DIR/${MP}_gate_done" ]; then
  echo "[step 3/8] Running augmented shortcut audit gate (trivial classifier must stay < 0.60 AUROC)..."
  set +e
  run_capped 3 "$PYTHON" -m src.eval \
    --shortcut-only \
    --augment-train-windows \
    --fail-on-shortcut \
    --manifest "$DATA_DIR/manifest.jsonl" \
    --max-per-slice "$GATE_MAX_PER_SLICE" \
    --results-dir "$RESULTS_DIR"
  gate_rc=$?
  set -e
  if [ "$gate_rc" -ne 0 ]; then
    if [ "$gate_rc" -eq 2 ]; then
      emit_gate_key
      echo "[step 3] GATE FAILED: trivial classifier still finds a shortcut on augmented training windows (see the [augmented audit] line above and $RESULTS_DIR/shortcut_audit_augmented_*.json). Stopping before training — no GPU training time spent."
      exit 2
    fi
    echo "[step 3] shortcut audit failed with rc=$gate_rc"
    exit "$gate_rc"
  fi
  emit_gate_key
  latest_audit="$(ls -t "$RESULTS_DIR"/shortcut_audit_augmented_*.json 2>/dev/null | head -1 || true)"
  if [ -n "$latest_audit" ]; then
    "$PYTHON" - "$latest_audit" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
mw = d.get("model_window_shortcuts", {})
aug = d.get("augmented_train_window_shortcuts", {})
print(f"[augmented audit] (from {sys.argv[1]}) trivial AUROC unaugmented={mw.get('auroc')} augmented={aug.get('auroc')} EER={aug.get('eer_pct')}%")
PY
  fi
  touch "$LOG_DIR/${MP}_gate_done"
else
  echo "[step 3/8] Shortcut gate already passed ($LOG_DIR/${MP}_gate_done) — skipping."
fi

# ---- step 4: train (per-epoch last.pt + HF upload + resume + deadline guard) --
if [ ! -f "$LOG_DIR/${MP}_train_done" ]; then
  echo "[step 4/8] Training (resume-from=${RESUME}, deadline_unix=${TRAIN_CMD_DEADLINE}, hf-prefix=${HF_PREFIX})..."
  train_args=(
    --manifest "$DATA_DIR/manifest.jsonl"
    --out-dir "$CKPT_DIR"
    --device "$DEVICE"
    --resume-from "$RESUME"
    --hf-prefix "$HF_PREFIX"
    --deadline-unix "$TRAIN_CMD_DEADLINE"
  )
  if [ "$DRY_RUN" != "1" ] && [ -n "$HF_MODEL_REPO" ]; then
    train_args+=(--hf-repo "$HF_MODEL_REPO")
  fi
  if [ -n "$MAX_STEPS_ARG" ]; then
    train_args+=(--max-steps "$MAX_STEPS_ARG")
  elif [ "$SMOKE" = "1" ] && [ -z "${MAX_STEPS:-}" ]; then
    train_args+=(--max-steps 20)
  elif [ -n "${MAX_STEPS:-}" ]; then
    train_args+=(--max-steps "$MAX_STEPS")
  fi
  if [ "$DRY_RUN" = "1" ]; then
    train_args+=(--tiny-test)
  fi
  set +e
  run_capped 4 "$PYTHON" -m src.train "${train_args[@]}"
  train_rc=$?
  set -e
  if [ "$train_rc" -eq 3 ]; then
    # In SMOKE the train deadline is min(MAX_HOURS train deadline, smoke cap):
    # rc 3 past the smoke deadline is a smoke cap stop (exit 5); otherwise it is
    # the session budget guard -> clean stop (exit 0).
    if [ "$SMOKE" = "1" ] && [ "$(date +%s)" -ge "$SMOKE_DEADLINE" ]; then
      smoke_stop 4
    fi
    budget_stop
  fi
  if [ "$train_rc" -ne 0 ]; then
    echo "[step 4] training failed with rc=$train_rc (last.pt checkpoint is resumable)"
    exit "$train_rc"
  fi
  touch "$LOG_DIR/${MP}_train_done"
else
  echo "[step 4/8] Training already completed ($LOG_DIR/${MP}_train_done) — skipping."
fi

check_budget

# ---- step 5: export ONNX (if needed) + INT8 parity ---------------------------
if [ ! -f "$LOG_DIR/${MP}_parity_done" ]; then
  echo "[step 5/8] Ensuring ONNX exports + running INT8 parity check..."
  if [ ! -f "$CKPT_DIR/cloakbreak_fp32.onnx" ] || [ ! -f "$CKPT_DIR/cloakbreak_int8.onnx" ]; then
    if [ ! -f "$CKPT_DIR/cloakbreak_best.pt" ]; then
      echo "[step 5] ERROR: $CKPT_DIR/cloakbreak_best.pt missing — nothing to export."
      exit 1
    fi
    export_args=(--export-only --out-dir "$CKPT_DIR")
    if [ "$DRY_RUN" = "1" ]; then
      export_args+=(--tiny-test)
    fi
    run_capped 5 "$PYTHON" -m src.train "${export_args[@]}"
  fi
  parity_args=(
    --int8-parity
    --manifest "$DATA_DIR/manifest.jsonl"
    --fp32-model "$CKPT_DIR/cloakbreak_fp32.onnx"
    --model-path "$CKPT_DIR/cloakbreak_int8.onnx"
    --max-auroc-diff "$INT8_PARITY_MAX_DIFF"
    --results-dir "$RESULTS_DIR"
  )
  if [ "$SMOKE" = "1" ]; then
    parity_args+=(--parity-max-clips 100)
  fi
  set +e
  run_capped 5 "$PYTHON" -m src.eval "${parity_args[@]}"
  parity_rc=$?
  set -e
  emit_int8_key
  if [ "$parity_rc" -eq 4 ]; then
    echo "[step 5] WARNING: INT8 parity FAILED (see $RESULTS_DIR/int8_parity_*.json) — falling back to fp32; INT8 will not be uploaded as a deployable artifact."
  elif [ "$parity_rc" -ne 0 ]; then
    echo "[step 5] INT8 parity check FAILED (rc=$parity_rc) — see $RESULTS_DIR/int8_parity_*.json"
    exit "$parity_rc"
  fi
  touch "$LOG_DIR/${MP}_parity_done"
else
  echo "[step 5/8] INT8 parity already passed ($LOG_DIR/${MP}_parity_done) — skipping."
fi

check_budget

# ---- step 6: full benchmark eval (fp32 torch checkpoint vs optional baseline) -
if [ ! -f "$LOG_DIR/${MP}_eval_done" ]; then
  echo "[step 6/8] Running full evaluation (fp32 torch${BASELINE_ID:+ vs $BASELINE_ID})..."
  eval_args=(
    --manifest "$DATA_DIR/manifest.jsonl"
    --backend torch
    --model-path "$CKPT_DIR/cloakbreak_best.pt"
    --device "$DEVICE"
    --max-per-slice "$EVAL_MAX_PER_SLICE"
    --out-json "$RESULTS_DIR/eval_results.json"
    --out-markdown "$RESULTS_DIR/results_table.md"
    --results-dir "$RESULTS_DIR"
  )
  if [ -n "$BASELINE_ID" ]; then
    eval_args+=(--baseline-id "$BASELINE_ID")
  fi
  run_capped 6 "$PYTHON" -m src.eval "${eval_args[@]}"
  echo "----- results_table.md -----"
  cat "$RESULTS_DIR/results_table.md"
  emit_headline_key
  touch "$LOG_DIR/${MP}_eval_done"
else
  echo "[step 6/8] Full eval already completed ($LOG_DIR/${MP}_eval_done) — skipping."
fi

check_budget

# SMOKE completed all capped steps — record it BEFORE upload so key_lines.txt
# (which lives in RESULTS_DIR) is uploaded together with the results.
if [ "$SMOKE" = "1" ]; then
  key_line "smoke $(smoke_min_fields) complete=true"
fi

# ---- step 7: upload artifacts to the private model repo ----------------------
if [ ! -f "$LOG_DIR/${MP}_upload_done" ]; then
  echo "[step 7/8] Uploading artifacts -> ${HF_MODEL_REPO} (prefix '${UPLOAD_PREFIX}')..."
  upload_args=(
    upload-artifacts
    --repo "$HF_MODEL_REPO"
    --ckpt-dir "$CKPT_DIR"
    --results-dir "$RESULTS_DIR"
    --log-dir "$LOG_DIR"
    --prefix "$UPLOAD_PREFIX"
  )
  if [ "$DRY_RUN" = "1" ]; then
    upload_args+=(--dry-run)
  fi
  # INT8 parity failed (rc 4 is non-fatal since we eval fp32): don't upload the
  # int8 ONNX as a deployable artifact.
  parity_json="$(ls -t "$RESULTS_DIR"/int8_parity_*.json 2>/dev/null | head -1 || true)"
  if [ -n "$parity_json" ]; then
    parity_passed="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('passed'))" "$parity_json" 2>/dev/null || true)"
    if [ "$parity_passed" = "False" ]; then
      upload_args+=(--exclude-int8)
      echo "[step 7] newest parity JSON reports passed=false — uploading with --exclude-int8"
    fi
  fi
  run_capped 7 "$PYTHON" -m src.hub "${upload_args[@]}"
  touch "$LOG_DIR/${MP}_upload_done"
else
  echo "[step 7/8] Artifact upload already completed ($LOG_DIR/${MP}_upload_done) — skipping."
fi

# ---- step 8: done ------------------------------------------------------------
INVOCATION_END="$(date +%s)"
INVOCATION_SEC=$(( INVOCATION_END - INVOCATION_START ))
SESSION_SEC=$(( INVOCATION_END - SESSION_START_AT ))
fmt_hms() {
  local s=$1
  printf "%d:%02d:%02d" $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}
echo "================================================================="
if [ "$SMOKE" = "1" ]; then
  echo "[session2] SMOKE COMPLETE — now run the full session: bash scripts/session2_vast.sh"
  echo "[smoke] used $(awk "BEGIN{printf \"%.1f\", ($(date +%s) - $SMOKE_CLOCK_START)/60.0}") of $(awk "BEGIN{printf \"%.1f\", $SMOKE_MAX_MIN}") budget minutes (setup steps 1–2: ${SMOKE_SETUP_MIN} min, shared with the full run)"
else
  echo "[session2] ALL STEPS COMPLETE"
fi
echo "[session2] This invocation: $(fmt_hms "$INVOCATION_SEC") | since session start: $(fmt_hms "$SESSION_SEC")"
if [ -n "$VAST_RATE_PER_HR" ]; then
  COST="$(awk "BEGIN{printf \"%.2f\", ($SESSION_SEC/3600.0)*$VAST_RATE_PER_HR}")"
  echo "[session2] Estimated cost since session start: \$${COST} at \$${VAST_RATE_PER_HR}/hr"
fi
echo "----- [KEY] lines ($KEY_LINES_FILE) -----"
if [ -f "$KEY_LINES_FILE" ]; then
  cat "$KEY_LINES_FILE"
fi
echo "[session2] DESTROY this Vast instance now (Destroy, not Stop)."
echo "================================================================="
