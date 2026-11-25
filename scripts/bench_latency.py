"""Measure per-window CPU latency of a CloakBreak ONNX model, as the demo app runs it.

Uses OnnxDetectorRuntime with its default thread count (same as app/main.py) on
real clips from a manifest split, after a short warmup. Latency is the
runtime's own latency_ms (loudness normalize + crop/pad + session.run + focus).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from src.audio_utils import TARGET_SR, load_wav_mono
from src.eval import _resolve_manifest_rows
from src.model import OnnxDetectorRuntime
from src.sysinfo import effective_cpu_count


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU latency benchmark for a CloakBreak ONNX model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [r for r in _resolve_manifest_rows(args.manifest) if r.get("split") == args.split][: args.clips]
    if not rows:
        raise SystemExit(f"no rows with split={args.split!r}")
    wavs = [load_wav_mono(r["path"], target_sr=TARGET_SR)[0] for r in rows]

    rt = OnnxDetectorRuntime(args.model_path)
    for w in wavs[: args.warmup]:
        rt.predict(w)
    lat = np.asarray([rt.predict(w).latency_ms for w in wavs], dtype=np.float64)

    out = {
        "model_path": str(args.model_path),
        "machine": platform.machine(),
        "effective_cpus": effective_cpu_count(),
        "onnx_intra_op_threads": int(rt.session.get_session_options().intra_op_num_threads),
        "window_sec": 2.5,
        "num_clips": int(lat.size),
        "warmup": args.warmup,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "latency_ms_mean": round(float(lat.mean()), 1),
    }
    print(json.dumps(out, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
