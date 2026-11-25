"""VAL-split threshold calibration for CloakBreak.

Selects the spoof decision threshold on the manifest ``val`` split ONLY — the
same rows, ``clean`` condition and ``seed=3000+i`` degradation as
``check_int8_parity`` so fp32 val AUROC is comparable to the Session-2 parity
number — then *reports* (never selects on) the same metrics on the test slices
rebuilt by ``eval.build_test_slice_rows``.

Scores are cached as JSONL per ``(backend tag, condition, seed base)`` so an
interrupted CPU run resumes without re-scoring.

CLI: ``python -m src.calibrate --manifest ... --backend onnx --model-path ...``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.audio_utils import TARGET_SR, load_wav_mono
from src.augment import EvalCondition, degrade_for_eval
from src.eval import (
    _resolve_manifest_rows,
    build_test_slice_rows,
    compute_eer_and_auroc,
)
from src.model import (
    DetectorRuntime,
    OnnxDetectorRuntime,
    TorchDetectorRuntime,
)
from src.sysinfo import effective_cpu_count

VAL_SEED_BASE = 3000  # matches check_int8_parity
TEST_SEED_BASE = 1000  # matches evaluate_slice
CLEAN_REAL_SEED_BASE = 2000  # matches evaluate_clean_real_fpr
EPS = 0.00005


def _rel_manifest_path(row: Dict, manifest_dir: Path) -> str:
    p = str(row.get("path", ""))
    try:
        rel = os.path.relpath(p, manifest_dir)
    except (OSError, ValueError):
        return p
    return rel if not rel.startswith("..") else p


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _cache_file(cache_dir: Path, tag: str, condition: str, seed_base: int) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    return Path(cache_dir) / f"{safe}__{condition}__s{seed_base}.jsonl"


def _load_score_cache(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out[str(d["path"])] = float(d["prob"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def _score_rows(
    runtime: DetectorRuntime,
    rows: List[Dict],
    condition: EvalCondition,
    seed_base: int,
    cache_path: Path,
    manifest_dir: Path,
    progress_every: int = 100,
) -> List[Dict]:
    """Score ``rows`` -> ``[{path,id,label,generator,prob}]`` in row order.

    Cached entries (keyed by manifest-relative path) are reused; new scores are
    appended to ``cache_path`` immediately so an interrupted run resumes.
    """
    cache = _load_score_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []
    total = len(rows)
    scored_new = 0
    t0 = time.perf_counter()
    step = max(1, int(progress_every))
    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            rel = _rel_manifest_path(row, manifest_dir)
            prob = cache.get(rel)
            if prob is None:
                wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
                degraded = degrade_for_eval(wav, condition=condition, sr=sr, seed=seed_base + i)
                prob = float(runtime.predict(degraded).spoof_prob)
                fh.write(json.dumps({"path": rel, "prob": prob}) + "\n")
                fh.flush()
                scored_new += 1
            out.append(
                {
                    "path": rel,
                    "id": row.get("id"),
                    "label": int(row["label"]),
                    "generator": str(row.get("generator", "")),
                    "prob": prob,
                }
            )
            done = i + 1
            if done % step == 0 or done == total:
                elapsed = max(time.perf_counter() - t0, 1e-6)
                rate = done / elapsed
                eta_min = (total - done) / max(rate, 1e-6) / 60.0
                print(
                    f"[calibrate] {done}/{total} clips ({rate:.1f} clips/s, "
                    f"{scored_new} newly scored, eta {eta_min:.1f} min)",
                    flush=True,
                )
    return out


def _jnum(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    return None if (f != f) else round(f, 6)


def _rate_at(scored: List[Dict], threshold: float) -> Dict:
    """FPR/TPR at ``prob >= threshold`` overall and per generator."""
    def _rate(group: List[Dict]) -> Optional[float]:
        if not group:
            return None
        return round(float(np.mean([1.0 if s["prob"] >= threshold else 0.0 for s in group])), 6)

    real = [s for s in scored if s["label"] == 0]
    spoof = [s for s in scored if s["label"] == 1]
    real_gens = sorted({s["generator"] for s in real})
    spoof_gens = sorted({s["generator"] for s in spoof})
    return {
        "threshold": threshold,
        "fpr": _rate(real),
        "tpr": _rate(spoof),
        "fpr_by_generator": {g: _rate([s for s in real if s["generator"] == g]) for g in real_gens},
        "tpr_by_generator": {g: _rate([s for s in spoof if s["generator"] == g]) for g in spoof_gens},
    }


def select_operating_threshold(real_probs_desc: Sequence[float], target_fpr: float) -> float:
    """``k = floor(target_fpr * n_real)``; ``t = real_desc[k] + EPS``.

    With ``prob >= t`` the at-most-k reals above ``real_desc[k]`` keep achieved
    val FPR <= target_fpr. ``k >= n_real`` falls back to the min real prob.
    """
    n = len(real_probs_desc)
    if n == 0:
        raise ValueError("cannot select a threshold with zero real val clips")
    k = math.floor(float(target_fpr) * n)
    if k >= n:
        return float(min(real_probs_desc))
    return float(real_probs_desc[k]) + EPS


def select_eer_threshold(scored: List[Dict]) -> Dict:
    """Threshold (``prob >= t``) over distinct val probs minimizing |FPR - FNR|."""
    real = np.asarray([s["prob"] for s in scored if s["label"] == 0], dtype=np.float64)
    spoof = np.asarray([s["prob"] for s in scored if s["label"] == 1], dtype=np.float64)
    if real.size == 0 or spoof.size == 0:
        raise ValueError("EER threshold needs both real and spoof val clips")
    best_t, best_d, best_fpr, best_fnr = 0.5, float("inf"), float("nan"), float("nan")
    for t in sorted(set(float(p) for p in np.concatenate([real, spoof]))):
        fpr = float(np.mean(real >= t))
        fnr = float(np.mean(spoof < t))
        d = abs(fpr - fnr)
        if d < best_d - 1e-12:
            best_t, best_d, best_fpr, best_fnr = t, d, fpr, fnr
    return {"threshold": best_t, "fpr": best_fpr, "fnr": best_fnr, "abs_diff": best_d}


def _metrics_block(scored: List[Dict]) -> Dict:
    labels = [s["label"] for s in scored]
    probs = [s["prob"] for s in scored]
    m = compute_eer_and_auroc(labels, probs)
    n_real = sum(1 for s in scored if s["label"] == 0)
    n_spoof = len(scored) - n_real
    return {
        "n_clips": len(scored),
        "n_real": n_real,
        "n_spoof": n_spoof,
        "eer_pct": _jnum(m["eer_pct"]),
        "auroc": _jnum(m["auroc"]),
    }


def _fmt4(v: Optional[float]) -> str:
    return "NA" if v is None else f"{float(v):.4f}"


def run_calibration(
    manifest_path: Path,
    runtime: DetectorRuntime,
    *,
    model_path: str,
    backend: str,
    cache_tag: str,
    device: str = "cpu",
    target_fpr: float = 0.05,
    max_per_slice: int = 500,
    cache_dir: Optional[Path] = None,
    parity_runtime: Optional[DetectorRuntime] = None,
    parity_model_path: Optional[str] = None,
    key_lines_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    progress_every: int = 100,
) -> Dict:
    """Score val+test slices, select the operating threshold on val, report."""
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    rows = _resolve_manifest_rows(manifest_path)
    val_rows = [
        r for r in rows
        if r.get("split") == "val" and "kokoro" not in str(r.get("generator", ""))
    ]
    if not val_rows:
        raise ValueError(
            f"calibrate requires manifest rows with split=='val' (non-Kokoro); {manifest_path} has none."
        )
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        test_rows = rows
    srows = build_test_slice_rows(test_rows, max_per_slice=max_per_slice)

    if cache_dir is None:
        cache_dir = Path("results") / "calibration_cache"
    cache_dir = Path(cache_dir)

    key_lines: List[str] = []

    def _emit(line: str) -> None:
        print(line, flush=True)
        key_lines.append(line)

    # ---- VAL (selection split) -------------------------------------------
    val_scored = _score_rows(
        runtime, val_rows, "clean", VAL_SEED_BASE,
        _cache_file(cache_dir, cache_tag, "clean", VAL_SEED_BASE),
        manifest_dir, progress_every=progress_every,
    )
    val_metrics = _metrics_block(val_scored)
    real_probs_desc = sorted(
        (s["prob"] for s in val_scored if s["label"] == 0), reverse=True
    )
    t_op = select_operating_threshold(real_probs_desc, target_fpr)
    eer_sel = select_eer_threshold(val_scored)
    t_eer = float(eer_sel["threshold"])

    val_block = dict(val_metrics)
    val_block.update(
        {
            "at_operating_threshold": _rate_at(val_scored, t_op),
            "at_eer_threshold": _rate_at(val_scored, t_eer),
            "at_0.5": _rate_at(val_scored, 0.5),
        }
    )

    _emit(
        f"[KEY] threshold op={t_op:.4f} val_fpr={_fmt4(val_block['at_operating_threshold']['fpr'])} "
        f"val_tpr={_fmt4(val_block['at_operating_threshold']['tpr'])} "
        f"val_eer_threshold={t_eer:.4f}"
    )

    # ---- TEST slices (report only) ----------------------------------------
    # unseen_kokoro_* pairs InTheWild reals with studio-clean Kokoro (channel-
    # confounded); clean_real_vs_kokoro_* (LibriSpeech vs Kokoro) is the honest
    # comparison — score and report both.
    test_specs = [
        ("clean_real", srows["clean_real_test"], "clean", CLEAN_REAL_SEED_BASE),
        ("test_clean", srows["main_pair"], "clean", TEST_SEED_BASE),
        ("test_opus_16k", srows["main_pair"], "opus_16k", TEST_SEED_BASE),
        ("unseen_kokoro_clean", srows["kokoro_all_pair"], "clean", TEST_SEED_BASE),
        ("unseen_kokoro_opus_16k", srows["kokoro_all_pair"], "opus_16k", TEST_SEED_BASE),
        ("clean_real_vs_kokoro_clean", srows["clean_vs_kokoro_pair"], "clean", TEST_SEED_BASE),
        ("clean_real_vs_kokoro_opus_16k", srows["clean_vs_kokoro_pair"], "opus_16k", TEST_SEED_BASE),
    ]
    test_block: Dict[str, Dict] = {}
    for name, srows_list, cond, seed_base in test_specs:
        if not srows_list:
            test_block[name] = {"skipped": "empty slice"}
            continue
        gen_counts: Dict[str, int] = {}
        for r in srows_list:
            g = str(r.get("generator", ""))
            gen_counts[g] = gen_counts.get(g, 0) + 1
        scored = _score_rows(
            runtime, srows_list, cond, seed_base,
            _cache_file(cache_dir, cache_tag, cond, seed_base),
            manifest_dir, progress_every=progress_every,
        )
        blk = _metrics_block(scored)
        blk["rows_by_generator"] = gen_counts
        blk.update(
            {
                "at_operating_threshold": _rate_at(scored, t_op),
                "at_eer_threshold": _rate_at(scored, t_eer),
                "at_0.5": _rate_at(scored, 0.5),
            }
        )
        test_block[name] = blk
        op_rates = blk["at_operating_threshold"]
        _emit(
            f"[KEY] threshold_test slice={name} fpr={_fmt4(op_rates['fpr'])} "
            f"tpr={_fmt4(op_rates['tpr'])}"
        )

    # ---- optional torch parity on VAL --------------------------------------
    parity_block: Optional[Dict] = None
    if parity_runtime is not None:
        ptag = cache_tag + "__parity"
        p_scored = _score_rows(
            parity_runtime, val_rows, "clean", VAL_SEED_BASE,
            _cache_file(cache_dir, ptag, "clean", VAL_SEED_BASE),
            manifest_dir, progress_every=progress_every,
        )
        p_probs = np.asarray([s["prob"] for s in p_scored], dtype=np.float64)
        v_probs = np.asarray([s["prob"] for s in val_scored], dtype=np.float64)
        diffs = np.abs(p_probs - v_probs)
        pm = compute_eer_and_auroc([s["label"] for s in p_scored], [s["prob"] for s in p_scored])
        parity_block = {
            "backend": "torch",
            "model_path": parity_model_path,
            "n_clips": len(p_scored),
            "primary": {"auroc": val_metrics["auroc"], "eer_pct": val_metrics["eer_pct"]},
            "parity": {"auroc": _jnum(pm["auroc"]), "eer_pct": _jnum(pm["eer_pct"])},
            "mean_abs_prob_diff": _jnum(float(diffs.mean())) if diffs.size else None,
            "max_abs_prob_diff": _jnum(float(diffs.max())) if diffs.size else None,
        }

    report = {
        "model_path": model_path,
        "model_sha256": _sha256_file(Path(model_path)) if Path(model_path).exists() else None,
        "backend": backend,
        "device": device,
        "target_fpr": float(target_fpr),
        "selection": "val split only; test never used for selection",
        "operating_threshold": t_op,
        "val": val_block,
        "val_eer_threshold": {
            "threshold": t_eer,
            "fpr": _jnum(eer_sel["fpr"]),
            "fnr": _jnum(eer_sel["fnr"]),
            "abs_fpr_fnr_diff": _jnum(eer_sel["abs_diff"]),
            "eer_pct": val_metrics["eer_pct"],
            "auroc": val_metrics["auroc"],
        },
        "test": test_block,
        "torch_parity": parity_block,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if key_lines_path is not None:
        kp = Path(key_lines_path)
        kp.parent.mkdir(parents=True, exist_ok=True)
        with open(kp, "a", encoding="utf-8") as fh:
            for line in key_lines:
                fh.write(line + "\n")

    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[calibrate] wrote {op}", flush=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate CloakBreak spoof threshold on the VAL split")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--max-per-slice", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("results/threshold_calibration.json"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Score cache dir (default: <out dir>/calibration_cache)")
    parser.add_argument("--torch-parity-ckpt", type=Path, default=None,
                        help="Also score VAL with this torch checkpoint (parity report only)")
    parser.add_argument("--onnx-threads", type=int, default=effective_cpu_count())
    parser.add_argument("--key-lines", type=Path, default=None,
                        help="Append [KEY] lines to this file (e.g. results/key_lines.txt)")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = args.out.parent / "calibration_cache"

    if args.backend == "onnx":
        runtime = OnnxDetectorRuntime(args.model_path, num_threads=args.onnx_threads)
    else:
        runtime = TorchDetectorRuntime(args.model_path, device=args.device)

    parity_runtime = None
    if args.torch_parity_ckpt is not None:
        parity_runtime = TorchDetectorRuntime(args.torch_parity_ckpt, device=args.device)

    run_calibration(
        args.manifest,
        runtime,
        model_path=str(args.model_path),
        backend=args.backend,
        cache_tag=f"{args.backend}-{args.model_path.name}",
        device=args.device,
        target_fpr=args.target_fpr,
        max_per_slice=args.max_per_slice,
        cache_dir=cache_dir,
        parity_runtime=parity_runtime,
        parity_model_path=str(args.torch_parity_ckpt) if args.torch_parity_ckpt else None,
        key_lines_path=args.key_lines,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
