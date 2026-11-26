"""Offline re-scorer for CLOAKBREAK_DEBUG_CAPTURE live sessions.

Loads a captured 16 kHz WAV plus its JSON sidecar (the windows the live
scorer emitted), re-scores every logged window on the exact samples
[audio_start_sec, audio_end_sec) with the same degrade_for_eval +
OnnxDetectorRuntime path the app's score_fn uses, prints a per-window
live-vs-offline table, and writes a JSON summary with --out.

Optionally scores a reference 16 kHz source WAV (the pre-browser audio) at
the same window positions, aligned by cross-correlating the first 10 s —
this is what scripts/bench_live_mic.py uses for its captured-vs-source
comparison.

Usage:
    python -m scripts.replay_capture --wav capture.wav --model-path model.onnx \
        [--json capture.json] [--out summary.json] [--reference-wav source16k.wav] \
        [--condition clean]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from src.augment import degrade_for_eval
from src.audio_utils import TARGET_SR, load_wav_mono
from src.model import OnnxDetectorRuntime

ALIGN_SECONDS = 10.0
MAX_ALIGN_SHIFT_SEC = 30.0


def load_sidecar_windows(json_path: str | Path) -> tuple[list[dict], dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    return list(data.get("windows", [])), data


def _window_bounds(start_sec: float, end_sec: float, n: int) -> tuple[int, int]:
    i0 = max(0, int(round(float(start_sec) * TARGET_SR)))
    i1 = min(n, int(round(float(end_sec) * TARGET_SR)))
    return i0, i1


def score_window(runtime, wav: np.ndarray, start_sec: float, end_sec: float, condition: str):
    """Score the exact [start, end) window the way the app's score_fn does."""
    i0, i1 = _window_bounds(start_sec, end_sec, wav.shape[0])
    seg = wav[i0:i1]
    degraded = degrade_for_eval(seg, condition=condition or "clean", sr=TARGET_SR)
    return runtime.predict(degraded)


def align_offset_samples(capture: np.ndarray, reference: np.ndarray) -> tuple[int, float]:
    """Return (offset_samples, peak_corr) such that capture[t] ~= reference[t + offset].

    Cross-correlates the first ALIGN_SECONDS of each signal over shifts up to
    MAX_ALIGN_SHIFT_SEC. offset > 0 means the capture started that many
    samples into the reference stream.
    """
    n = min(int(ALIGN_SECONDS * TARGET_SR), capture.shape[0], reference.shape[0])
    if n < TARGET_SR:
        return 0, 0.0
    cap = capture[:n].astype(np.float64)
    max_shift = min(int(MAX_ALIGN_SHIFT_SEC * TARGET_SR), reference.shape[0] - n)
    if max_shift <= 0:
        return 0, 0.0
    ref = reference[: n + max_shift].astype(np.float64)
    # correlate(ref, cap, 'valid')[s] = sum_t ref[s + t] * cap[t]
    corr = correlate(ref, cap, mode="valid", method="fft")
    # Normalize by local window energy so loudness differences don't dominate.
    cs = np.concatenate([[0.0], np.cumsum(ref * ref)])
    e = cs[n : n + max_shift + 1] - cs[: max_shift + 1]
    denom = np.sqrt(np.maximum(e[: corr.shape[0]], 1e-12) * float(np.dot(cap, cap)))
    score = corr[: denom.shape[0]] / denom
    best = int(np.argmax(score))
    return best, float(score[best])


def replay_capture(
    wav_path: str | Path,
    model_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    condition: str = "clean",
    reference_wav: str | Path | None = None,
    runtime=None,
) -> dict:
    """Re-score a captured session. Returns the summary dict (also printed)."""
    wav, sr = load_wav_mono(wav_path, target_sr=TARGET_SR)
    if runtime is None:
        if not model_path:
            raise ValueError("model_path is required when runtime is not provided")
        runtime = OnnxDetectorRuntime(model_path)

    windows, sidecar = [], {}
    if sidecar_path:
        windows, sidecar = load_sidecar_windows(sidecar_path)

    threshold = None
    for w in windows:
        if w.get("threshold") is not None:
            threshold = float(w["threshold"])
            break
    threshold = 0.5 if threshold is None else threshold

    rows = []
    diffs = []
    for idx, w in enumerate(windows):
        live_prob = w.get("spoof_prob")
        cond = w.get("condition") or "clean"
        try:
            res = score_window(runtime, wav, w["audio_start_sec"], w["audio_end_sec"], cond)
            offline_prob = float(res.spoof_prob)
        except Exception as exc:  # keep going — one bad window shouldn't kill the report
            offline_prob = None
            rows.append({
                "idx": idx,
                "audio_start_sec": w.get("audio_start_sec"),
                "audio_end_sec": w.get("audio_end_sec"),
                "condition": cond,
                "live_prob": live_prob,
                "offline_prob": None,
                "error": str(exc),
            })
            continue
        diff = abs(offline_prob - float(live_prob)) if live_prob is not None else None
        if diff is not None:
            diffs.append(diff)
        rows.append({
            "idx": idx,
            "audio_start_sec": w.get("audio_start_sec"),
            "audio_end_sec": w.get("audio_end_sec"),
            "condition": cond,
            "live_prob": live_prob,
            "offline_prob": offline_prob,
            "abs_diff": diff,
            "offline_ge_threshold": offline_prob >= threshold,
        })

    # Upload-path score: what /api/detect-file gives — degrade the (<=20 s)
    # clip, then predict() center-crops to the 2.5 s window.
    upload = {"condition": condition, "skipped": "no audio"}
    clip = wav[: TARGET_SR * 20]
    if clip.shape[0] / float(TARGET_SR) >= 2.0:
        degraded = degrade_for_eval(clip, condition=condition, sr=TARGET_SR)
        res = runtime.predict(degraded)
        upload = {
            "condition": condition,
            "spoof_prob": res.spoof_prob,
            "verdict": "spoof" if res.spoof_prob >= threshold else "bonafide",
            "latency_ms": res.latency_ms,
            "backend": res.backend,
            "duration_s": round(float(degraded.shape[0]) / TARGET_SR, 2),
        }

    summary = {
        "wav": str(wav_path),
        "sidecar": str(sidecar_path) if sidecar_path else None,
        "model_path": str(model_path) if model_path else getattr(runtime, "backend_name", None),
        "threshold": threshold,
        "n_windows": len(rows),
        "windows": rows,
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "frac_offline_ge_threshold": (
            float(np.mean([r["offline_ge_threshold"] for r in rows if "offline_ge_threshold" in r]))
            if any("offline_ge_threshold" in r for r in rows) else None
        ),
        "upload_path": upload,
        "sidecar_meta": {k: v for k, v in sidecar.items() if k != "windows"},
    }

    if reference_wav:
        summary["reference"] = _score_reference(runtime, wav, reference_wav, rows, threshold)

    return summary


def _score_reference(runtime, capture: np.ndarray, reference_wav, rows, threshold) -> dict:
    ref, _sr = load_wav_mono(reference_wav, target_sr=TARGET_SR)
    offset, peak = align_offset_samples(capture, ref)
    offset_sec = offset / float(TARGET_SR)
    out_rows = []
    diffs = []
    ge = []
    for r in rows:
        if "audio_start_sec" not in r or r.get("offline_prob") is None:
            continue
        a = r["audio_start_sec"] + offset_sec
        b = r["audio_end_sec"] + offset_sec
        i0, i1 = _window_bounds(a, b, ref.shape[0])
        if i1 - i0 < int(0.5 * TARGET_SR):
            continue
        seg = ref[i0:i1]
        try:
            res = runtime.predict(degrade_for_eval(seg, condition=r.get("condition") or "clean", sr=TARGET_SR))
            src_prob = float(res.spoof_prob)
        except Exception as exc:
            out_rows.append({"idx": r["idx"], "error": str(exc)})
            continue
        d = abs(src_prob - float(r["offline_prob"]))
        diffs.append(d)
        ge.append(src_prob >= threshold)
        out_rows.append({
            "idx": r["idx"],
            "source_prob": src_prob,
            "abs_diff_vs_capture": d,
            "source_ge_threshold": src_prob >= threshold,
        })
    return {
        "wav": str(reference_wav),
        "offset_samples": int(offset),
        "offset_sec": round(offset_sec, 4),
        "align_peak": round(peak, 4),
        "n_scored": len(diffs),
        "frac_source_ge_threshold": float(np.mean(ge)) if ge else None,
        "mean_abs_diff_capture_vs_source": float(np.mean(diffs)) if diffs else None,
        "max_abs_diff_capture_vs_source": float(np.max(diffs)) if diffs else None,
        "windows": out_rows,
    }


def _print_table(summary: dict) -> None:
    print(f"wav: {summary['wav']}")
    if summary.get("sidecar"):
        print(f"sidecar: {summary['sidecar']}")
    print(f"model: {summary['model_path']}  threshold: {summary['threshold']:.4f}")
    print(f"{'idx':>4} {'start_s':>8} {'end_s':>8} {'cond':<11} {'live':>8} {'offline':>8} {'diff':>8}  flags")
    for r in summary["windows"]:
        if r.get("offline_prob") is None:
            print(f"{r['idx']:>4} {r.get('audio_start_sec')!s:>8} {r.get('audio_end_sec')!s:>8} "
                  f"{r.get('condition', ''):<11} {r.get('live_prob')!s:>8} {'ERR':>8} {'-':>8}  {r.get('error', '')}")
            continue
        flags = []
        if r.get("offline_ge_threshold"):
            flags.append(">=t")
        print(f"{r['idx']:>4} {r['audio_start_sec']:>8.3f} {r['audio_end_sec']:>8.3f} "
              f"{r.get('condition', ''):<11} {r.get('live_prob')!s:>8} {r['offline_prob']:>8.4f} "
              f"{r.get('abs_diff')!s:>8}  {' '.join(flags)}")
    print(f"\nwindows: {summary['n_windows']}  "
          f"max|live-offline|: {summary['max_abs_diff']}  "
          f"mean|live-offline|: {summary['mean_abs_diff']}")
    print(f"frac offline >= threshold: {summary['frac_offline_ge_threshold']}")
    up = summary["upload_path"]
    if "spoof_prob" in up:
        print(f"upload path ({up['condition']}): spoof_prob={up['spoof_prob']} "
              f"verdict={up['verdict']} latency_ms={up['latency_ms']} backend={up['backend']}")
    else:
        print(f"upload path: skipped ({up.get('skipped')})")
    ref = summary.get("reference")
    if ref:
        print(f"\nreference {ref['wav']}: offset={ref['offset_samples']} samples "
              f"({ref['offset_sec']} s, align peak {ref['align_peak']})")
        print(f"  source windows scored: {ref['n_scored']}  "
              f"frac source >= threshold: {ref['frac_source_ge_threshold']}")
        print(f"  mean|capture-source|: {ref['mean_abs_diff_capture_vs_source']}  "
              f"max: {ref['max_abs_diff_capture_vs_source']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True, help="Captured 16 kHz WAV")
    ap.add_argument("--model-path", default=None, help="ONNX model path")
    ap.add_argument("--json", dest="sidecar", default=None, help="Sidecar JSON with logged windows")
    ap.add_argument("--out", default=None, help="Write JSON summary to this path")
    ap.add_argument("--condition", default="clean", help="Condition for the upload-path score")
    ap.add_argument("--reference-wav", default=None,
                    help="Optional 16 kHz reference WAV to score at the same window positions")
    args = ap.parse_args()

    summary = replay_capture(
        args.wav,
        model_path=args.model_path,
        sidecar_path=args.sidecar,
        condition=args.condition,
        reference_wav=args.reference_wav,
    )
    _print_table(summary)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
