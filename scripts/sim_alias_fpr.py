"""Measure how the old naive browser decimation affects false alarms on real speech.

For known-real clips (default: LibriSpeech test, generator bonafide_clean), the
16 kHz audio is upsampled to 48 kHz (what a browser mic delivers), optional
band-limited 8.5-20 kHz noise is added to stand in for out-of-band mic energy,
and the result is brought back to 16 kHz two ways:

  naive  : x48[::3]                    (the removed downsampleTo16k behaviour)
  proper : resample_poly(x48, 1, 3)    (band-limited, like the browser / FIR path)

Each version is scored with the fp32 ONNX detector exactly like /api/detect-file
(center 2.5 s window). Reports mean score and the fraction at or above the
calibrated threshold for every (noise level, path) pair.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from src.audio_utils import TARGET_SR, load_wav_mono
from src.eval import _resolve_manifest_rows
from src.model import OnnxDetectorRuntime


def hf_noise(n: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    sos = butter(8, [8500, 20000], btype="bandpass", fs=48000, output="sos")
    noise = sosfilt(sos, rng.standard_normal(n))
    return noise * (rms / max(float(np.sqrt(np.mean(noise**2))), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--generator", default="bonafide_clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=120)
    ap.add_argument("--noise-dbfs", type=float, nargs="*", default=[-45.0, -35.0])
    ap.add_argument("--threshold", type=float, default=0.95255)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [
        r for r in _resolve_manifest_rows(args.manifest)
        if r.get("split") == args.split and r.get("generator") == args.generator
    ]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.clips]
    rt = OnnxDetectorRuntime(args.model_path)
    rng = np.random.default_rng(args.seed)

    levels = [None] + list(args.noise_dbfs)
    scores = {f"{lvl}|{path}": [] for lvl in levels for path in ("naive", "proper")}
    for i, r in enumerate(rows):
        x16, _ = load_wav_mono(r["path"], target_sr=TARGET_SR)
        x48 = resample_poly(x16.astype(np.float64), 3, 1)
        for lvl in levels:
            y48 = x48 if lvl is None else x48 + hf_noise(x48.size, 10.0 ** (lvl / 20.0), rng)
            y48 = np.clip(y48, -1.0, 1.0)
            versions = {
                "naive": y48[::3],
                "proper": resample_poly(y48, 1, 3),
            }
            for path, y16 in versions.items():
                scores[f"{lvl}|{path}"].append(float(rt.predict(y16.astype(np.float32)).spoof_prob))
        if (i + 1) % 20 == 0:
            print(f"[sim] {i + 1}/{len(rows)} clips", flush=True)

    summary = {}
    for key, vals in scores.items():
        lvl, path = key.split("|")
        a = np.asarray(vals)
        summary[key] = {
            "hf_noise_dbfs": None if lvl == "None" else float(lvl),
            "path": path,
            "n_clips": int(a.size),
            "mean_score": round(float(a.mean()), 4),
            "p95_score": round(float(np.percentile(a, 95)), 4),
            "frac_ge_threshold": round(float(np.mean(a >= args.threshold)), 4),
        }
    report = {
        "manifest": str(args.manifest),
        "model_path": str(args.model_path),
        "generator": args.generator,
        "split": args.split,
        "seed": args.seed,
        "threshold": args.threshold,
        "results": summary,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
