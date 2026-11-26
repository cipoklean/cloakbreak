"""Re-score a CLOAKBREAK_DEBUG_CAPTURE WAV on a fixed window grid, ignoring the live gate.

The live app only scores windows its speech gate lets through, so a capture
where the gate rejected everything has no live scores at all. This script
scores every 2.5 s window on a fixed hop (default 0.5 s) with the same
degrade_for_eval + runtime path the app uses, records each window's RMS level,
and replays the capture through ``SpeechGate`` in 1024-sample chunks (the
browser's chunk size at 16 kHz) to report which grid windows the live gate
would score.

Usage:
    python -m scripts.rescore_capture_grid --wav results/live_capture/<file>.wav \
        --model-path artifacts/session2/onnx/cloakbreak_fp32.onnx \
        [--baseline-id Vansh180/deepfake-audio-wav2vec2] [--out summary.json]
        [--scores-json previous_summary.json]   # reuse scores, recompute gate only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.audio_utils import TARGET_SR, SpeechGate, load_wav_mono, rms_dbfs, rms_vad

WINDOW_SEC = 2.5
CHUNK = 1024


def grid_starts(n_samples: int, hop_sec: float, window_sec: float = WINDOW_SEC) -> list[float]:
    dur = n_samples / float(TARGET_SR)
    starts, s = [], 0.0
    while s + window_sec <= dur + 1e-9:
        starts.append(round(s, 3))
        s += hop_sec
    return starts


def gate_decisions(wav: np.ndarray, starts: list[float], window_sec: float = WINDOW_SEC) -> list[dict]:
    """Stream the capture through SpeechGate; decide each grid window when its last sample arrives."""
    win = int(round(window_sec * TARGET_SR))
    ends = sorted((int(round(s * TARGET_SR)) + win, i) for i, s in enumerate(starts))
    out: list[dict | None] = [None] * len(starts)
    gate = SpeechGate(sr=TARGET_SR)
    pos, k = 0, 0
    while k < len(ends):
        end, i = ends[k]
        if pos >= end:
            seg = wav[end - win : end]
            out[i] = {
                "gate_speech": bool(gate.is_speech(seg)),
                "speech_gate_dbfs": round(gate.gate_dbfs(), 1),
                "noise_floor_dbfs": None if gate.noise_floor_dbfs() is None else round(gate.noise_floor_dbfs(), 1),
                "old_rms_vad_speech": bool(rms_vad(seg)),
            }
            k += 1
            continue
        step = min(CHUNK, end - pos)
        gate.push(wav[pos : pos + step])
        pos += step
    return out  # type: ignore[return-value]


def score_grid(wav: np.ndarray, starts: list[float], model_path: str, baseline_id: str | None) -> list[dict]:
    from src.augment import degrade_for_eval
    from src.model import BaselineDetectorRuntime, OnnxDetectorRuntime

    rt = OnnxDetectorRuntime(model_path)
    bl = BaselineDetectorRuntime(baseline_id, device="cpu") if baseline_id else None
    win = int(round(WINDOW_SEC * TARGET_SR))
    rows = []
    for s in starts:
        i0 = int(round(s * TARGET_SR))
        seg = wav[i0 : i0 + win]
        degraded = degrade_for_eval(seg, condition="clean", sr=TARGET_SR)
        row = {"start": s, "spoof": round(float(rt.predict(degraded).spoof_prob), 4), "dbfs": round(rms_dbfs(seg), 1)}
        if bl is not None:
            row["baseline"] = round(float(bl.predict_prob(degraded)), 4)
        rows.append(row)
    return rows


def summarize(rows: list[dict], threshold: float) -> dict:
    sp = np.array([r["spoof"] for r in rows])
    gated = np.array([r["gate_speech"] for r in rows])
    summary = {
        "n_windows": len(rows),
        "n_ge_threshold": int((sp >= threshold).sum()),
        "frac_ge_threshold": round(float((sp >= threshold).mean()), 4),
        "median_spoof": round(float(np.median(sp)), 4),
        "old_rms_vad_n_speech": int(sum(r["old_rms_vad_speech"] for r in rows)),
        "gate_n_speech": int(gated.sum()),
        "gate_n_ge_threshold": int(((sp >= threshold) & gated).sum()),
    }
    if gated.any():
        summary["gate_frac_ge_threshold"] = round(float((sp[gated] >= threshold).mean()), 4)
    if all("baseline" in r for r in rows):
        summary["baseline_n_ge_0.5"] = int(sum(r["baseline"] >= 0.5 for r in rows))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model-path", default="artifacts/session2/onnx/cloakbreak_fp32.onnx")
    ap.add_argument("--baseline-id", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="default: results/threshold_calibration.json")
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--scores-json", default=None, help="reuse per-window scores from an earlier --out file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    threshold = args.threshold
    if threshold is None:
        cal = Path("results/threshold_calibration.json")
        threshold = float(json.loads(cal.read_text(encoding="utf-8"))["operating_threshold"])

    wav, _ = load_wav_mono(args.wav, target_sr=TARGET_SR)
    starts = grid_starts(wav.shape[0], args.hop_sec)
    if args.scores_json:
        prev = json.loads(Path(args.scores_json).read_text(encoding="utf-8"))
        rows = [{k: r[k] for k in ("start", "spoof", "dbfs", "baseline") if k in r} for r in prev["rows"]]
        if [r["start"] for r in rows] != starts:
            raise SystemExit("--scores-json grid does not match this WAV and --hop-sec")
    else:
        rows = score_grid(wav, starts, args.model_path, args.baseline_id)
    for row, g in zip(rows, gate_decisions(wav, starts)):
        row.update(g)

    result = {
        "wav": Path(args.wav).name,
        "duration_s": round(wav.shape[0] / float(TARGET_SR), 2),
        "overall_dbfs": round(rms_dbfs(wav), 1),
        "window_sec": WINDOW_SEC,
        "hop_sec": args.hop_sec,
        "condition": "clean",
        "threshold": threshold,
        "summary": summarize(rows, threshold),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
