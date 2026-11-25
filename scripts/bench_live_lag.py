"""End-to-end live-lag benchmark for the /ws/detect demo.

Streams real manifest ``split=="test"`` audio through a real uvicorn app at
real time (or ``--speed``) and measures client-observed lag versus the server's
own ``lag_ms``, plus LiveScorer stats. Run on the target VM:

    python -m scripts.bench_live_lag \
        --manifest data/cloakbreak_v1/manifest.jsonl \
        --model-path checkpoints/cloakbreak_fp32.onnx \
        --duration 120 --out results/live_lag.json
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import contextlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.audio_utils import TARGET_SR, float32_to_pcm16_bytes, load_wav_mono
from src.eval import _resolve_manifest_rows
from src.sysinfo import effective_cpu_count

REPO_ROOT = Path(__file__).resolve().parents[1]


def send_time_for_audio_end(
    audio_end_sec: float,
    sr: int,
    chunk_end_indices: Sequence[int],
    send_times: Sequence[float],
) -> Optional[float]:
    """Monotonic send time of the first chunk whose end index >= audio_end_sec*sr (within rounding).

    ``chunk_end_indices[i]`` is the exclusive end sample index of chunk ``i``
    (i.e. samples sent after it); ``send_times[i]`` its send timestamp.
    """
    if not send_times:
        return None
    # The server rounds audio_end_sec to 3 decimals (error <= 0.5 ms), so allow
    # that much slack; otherwise a value rounded up would match the NEXT chunk
    # and understate client lag by up to one chunk.
    tol = int(math.ceil(0.0005 * sr))
    end_idx = int(round(float(audio_end_sec) * sr)) - tol
    i = bisect.bisect_left(list(chunk_end_indices), end_idx)
    i = min(max(i, 0), len(send_times) - 1)
    return float(send_times[i])


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 1)


def _summarize_lags(client_lags: List[float], server_lags: List[float], inf_ms: List[float]) -> Dict:
    return {
        "client_lag_ms": {
            "p50": _pct(client_lags, 50),
            "p95": _pct(client_lags, 95),
            "max": round(max(client_lags), 1) if client_lags else None,
        },
        "server_lag_ms": {
            "p50": _pct(server_lags, 50),
            "p95": _pct(server_lags, 95),
            "max": round(max(server_lags), 1) if server_lags else None,
        },
        "inference_ms": {"p50": _pct(inf_ms, 50), "p95": _pct(inf_ms, 95)},
    }


def _build_stream(manifest: Path, duration_s: float, seed: int) -> np.ndarray:
    rows = [r for r in _resolve_manifest_rows(manifest) if r.get("split") == "test"]
    if not rows:
        raise SystemExit(f"no split=='test' rows in {manifest}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    need = int(round(duration_s * TARGET_SR))
    parts: List[np.ndarray] = []
    total = 0
    for r in rows:
        wav, _sr = load_wav_mono(r["path"], target_sr=TARGET_SR)
        parts.append(wav)
        total += int(wav.size)
        if total >= need:
            break
    if total < need:
        print(f"[bench] WARN: stream is {total / TARGET_SR:.1f}s < requested {duration_s}s", file=sys.stderr)
    return np.concatenate(parts, axis=0)[:need].astype(np.float32)


def _start_uvicorn(model_path: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "CLOAKBREAK_BACKEND": "onnx",
            "CLOAKBREAK_MODEL_PATH": str(model_path),
            "ENABLE_CLONE_ENDPOINT": "false",
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 120.0) -> Dict:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"uvicorn exited early (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - poll loop
            last_err = exc
            time.sleep(0.5)
    raise SystemExit(f"/api/health did not come up within {timeout_s}s (last error: {last_err})")


async def _run_client(
    port: int,
    pcm_blobs: List[bytes],
    chunk_end_indices: List[int],
    sr: int,
    speed: float,
    tail_wait_s: float,
) -> Tuple[List[float], List[Tuple[float, Dict]]]:
    from websockets.asyncio.client import connect

    recv_log: List[Tuple[float, Dict]] = []
    stats_seen = asyncio.Event()

    async def receiver(ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            recv_log.append((time.monotonic(), msg))
            if msg.get("type") == "stats":
                stats_seen.set()

    async with connect(f"ws://127.0.0.1:{port}/ws/detect", max_size=None) as ws:
        await ws.send(json.dumps({"condition": "clean"}))
        rtask = asyncio.create_task(receiver(ws))
        try:
            send_times: List[float] = []
            t0 = time.monotonic()
            for blob, end_idx in zip(pcm_blobs, chunk_end_indices):
                target = t0 + end_idx / float(sr) / speed
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(blob)
                send_times.append(time.monotonic())

            await asyncio.sleep(tail_wait_s)
            await ws.send(json.dumps({"type": "stats"}))
            try:
                await asyncio.wait_for(stats_seen.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[bench] WARN: no stats reply within 30s", file=sys.stderr)
        finally:
            rtask.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rtask
    return send_times, recv_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Live websocket lag benchmark for CloakBreak")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120.0, help="stream seconds")
    parser.add_argument("--chunk-samples", type=int, default=1365)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stream = _build_stream(args.manifest, args.duration, args.seed)
    cs = int(args.chunk_samples)
    pcm_blobs = [
        float32_to_pcm16_bytes(stream[i : i + cs])
        for i in range(0, stream.size, cs)
    ]
    chunk_end_indices = [min((i + 1) * cs, stream.size) for i in range(len(pcm_blobs))]
    print(f"[bench] stream {stream.size / TARGET_SR:.1f}s in {len(pcm_blobs)} chunks of {cs} samples")

    proc = _start_uvicorn(args.model_path, args.port)
    try:
        health = _wait_health(args.port, proc)
        print(f"[bench] health: {json.dumps(health)}")
        send_times, recv_log = asyncio.run(
            _run_client(args.port, pcm_blobs, chunk_end_indices, TARGET_SR, args.speed, tail_wait_s=5.0)
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    speech: List[Tuple[float, Dict]] = []
    nonspeech = 0
    stats_msg: Optional[Dict] = None
    for t, msg in recv_log:
        if msg.get("type") == "stats":
            stats_msg = msg
            continue
        if msg.get("type") != "prediction":
            continue
        if msg.get("is_speech") is False:
            nonspeech += 1
            continue
        if msg.get("spoof_prob") is None:
            continue  # warming_up
        speech.append((t, msg))

    client_lags: List[float] = []
    server_lags: List[float] = []
    inf_ms: List[float] = []
    intervals: List[float] = []
    prev_t: Optional[float] = None
    for t, msg in speech:
        st = send_time_for_audio_end(msg.get("audio_end_sec", 0.0), TARGET_SR, chunk_end_indices, send_times)
        if st is not None:
            client_lags.append((t - st) * 1000.0)
        if msg.get("lag_ms") is not None:
            server_lags.append(float(msg["lag_ms"]))
        if msg.get("inference_ms") is not None:
            inf_ms.append(float(msg["inference_ms"]))
        if prev_t is not None:
            intervals.append((t - prev_t) * 1000.0)
        prev_t = t

    half = len(client_lags) // 2
    first_half = client_lags[:half] if half else []
    second_half = client_lags[half:] if half else []
    tail = _summarize_lags(client_lags[1:], server_lags[1:], inf_ms[1:])

    report = {
        "machine": platform.machine(),
        "effective_cpus": effective_cpu_count(),
        "model_path": str(args.model_path),
        "health": health,
        "duration_s": args.duration,
        "chunk_samples": cs,
        "speed": args.speed,
        "seed": args.seed,
        "n_speech_verdicts": len(speech),
        "n_nonspeech": nonspeech,
        "verdict_interval_ms": {"p50": _pct(intervals, 50), "p95": _pct(intervals, 95)},
        "client_lag_ms": {
            "p50": _pct(client_lags, 50),
            "p95": _pct(client_lags, 95),
            "max": round(max(client_lags), 1) if client_lags else None,
        },
        "server_lag_ms": {
            "p50": _pct(server_lags, 50),
            "p95": _pct(server_lags, 95),
            "max": round(max(server_lags), 1) if server_lags else None,
        },
        "inference_ms": {"p50": _pct(inf_ms, 50), "p95": _pct(inf_ms, 95)},
        "first_half_client_lag_p95_ms": _pct(first_half, 95),
        "second_half_client_lag_p95_ms": _pct(second_half, 95),
        "excluding_first_verdict": tail,
        "stats": stats_msg,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
