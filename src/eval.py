from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize

from src.audio_utils import (
    TARGET_SR,
    audio_rms,
    estimate_snr_db,
    load_wav_mono,
    normalize_loudness,
    pad_or_crop,
    spectral_rolloff_hz,
)
from src.augment import (
    SPECTRAL_AUG_RANGES,
    EvalCondition,
    TrainCodecAugmenter,
    degrade_for_eval,
    has_ffmpeg,
    precompute_opus_cache,
)
from src.generate_spoofs import assert_disjoint_text_splits
from src.model import (
    DEFAULT_BASELINE_ID,
    DEFAULT_WINDOW_SAMPLES,
    BaselineDetectorRuntime,
    DetectorRuntime,
    MockDetectorRuntime,
    OnnxDetectorRuntime,
    TorchDetectorRuntime,
)

SHORTCUT_FEATURE_NAMES = ["duration_s", "rms", "est_snr_db", "rolloff_85_hz"]


def compute_eer_and_auroc(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return {"eer_pct": float("nan"), "auroc": float("nan"), "f1_at_05": float("nan"), "eer_threshold": 0.5}

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    # Group tied scores (matches sklearn.metrics.roc_curve) so constant scores yield AUROC=0.5, EER=50%
    distinct_idx = np.where(np.diff(s_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]
    tpr = tps / float(pos.size)
    fpr = fps / float(neg.size)
    thresh_vals = s_sorted[threshold_idxs]

    tpr_curve = np.concatenate([[0.0], tpr])
    fpr_curve = np.concatenate([[0.0], fpr])
    auroc = float(np.trapz(tpr_curve, fpr_curve))

    fnr_curve = 1.0 - tpr_curve
    diff = fpr_curve - fnr_curve
    idx = int(np.argmin(np.abs(diff)))
    if diff[idx] == 0.0:
        eer = float(fpr_curve[idx])
    else:
        cross = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        if cross.size > 0:
            i0 = int(cross[0])
            d0, d1 = abs(diff[i0]), abs(diff[i0 + 1])
            alpha = d0 / (d0 + d1 + 1e-12)
            eer = float(fpr_curve[i0] + alpha * (fpr_curve[i0 + 1] - fpr_curve[i0]))
        else:
            eer = float(0.5 * (fpr_curve[idx] + fnr_curve[idx]))
    eer_thresh = float(thresh_vals[min(max(idx - 1, 0), thresh_vals.size - 1)])

    preds = (s >= 0.5).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "eer_pct": round(eer * 100.0, 2),
        "auroc": round(auroc, 4),
        "f1_at_05": round(f1, 4),
        "eer_threshold": round(eer_thresh, 4),
    }


def _fit_logistic_weights(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    constant_cols = sigma < 1e-4
    sigma_safe = np.where(constant_cols, 1.0, sigma)
    xtr = np.where(constant_cols, 0.0, (x_train - mu) / sigma_safe)
    xtr_b = np.hstack([xtr, np.ones((xtr.shape[0], 1))])

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(xtr_b @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        reg = 1e-2 * float(np.sum(w[:-1] ** 2))
        loss = -float(np.mean(y_train * np.log(p + 1e-12) + (1.0 - y_train) * np.log(1.0 - p + 1e-12))) + reg
        grad = (xtr_b.T @ (p - y_train)) / float(len(y_train))
        grad[:-1] += 2e-2 * w[:-1]
        return loss, grad

    res = minimize(loss_and_grad, np.zeros(xtr_b.shape[1]), jac=True, method="BFGS")
    return res.x, mu, sigma_safe, constant_cols


def _score_logistic(
    w: np.ndarray,
    mu: np.ndarray,
    sigma_safe: np.ndarray,
    constant_cols: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    xte = np.where(constant_cols, 0.0, (x_test - mu) / sigma_safe)
    xte_b = np.hstack([xte, np.ones((xte.shape[0], 1))])
    probs = 1.0 / (1.0 + np.exp(-np.clip(xte_b @ w, -30.0, 30.0)))
    return compute_eer_and_auroc(y_test.astype(np.int32).tolist(), probs.tolist())


def _resolve_manifest_rows(manifest_path: Path) -> List[Dict]:
    base = manifest_path.parent
    out: List[Dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = Path(row["path"])
        if not p.is_absolute():
            row["path"] = str((base / p).resolve())
        out.append(row)
    return out


def _is_shortcut(m: Dict[str, float]) -> bool:
    auc = m["auroc"]
    eer = m["eer_pct"]
    if np.isnan(auc) or np.isnan(eer):
        return False
    return (auc >= 0.60) or (eer <= 40.0)


def _feature_means(feats: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls_name, cls_val in (("real", 0), ("spoof", 1)):
        sub = feats[labels == cls_val]
        if sub.size == 0:
            continue
        out[cls_name] = {
            name: round(float(sub[:, i].mean()), 4) for i, name in enumerate(SHORTCUT_FEATURE_NAMES)
        }
    return out


def _select_audit_rows(rows: List[Dict], max_per_split: int) -> tuple[List[Dict], List[Dict]]:
    """Shared train/test row selection for shortcut audits (train excludes Kokoro)."""
    train_rows = [r for r in rows if r.get("split") == "train" and "kokoro" not in r.get("generator", "")][:max_per_split]
    test_rows = [r for r in rows if r.get("split") == "test"][:max_per_split]
    if not train_rows or not test_rows:
        mid = max(2, len(rows) // 2)
        train_rows, test_rows = rows[:mid], rows[mid:]
    return train_rows, test_rows


def _clean_model_window(wav: np.ndarray, sr: int) -> np.ndarray:
    return normalize_loudness(pad_or_crop(wav, DEFAULT_WINDOW_SAMPLES, random_crop=False))


def _window_shortcut_features(raw_dur_s: float, win: np.ndarray) -> List[float]:
    win_sec = DEFAULT_WINDOW_SAMPLES / float(TARGET_SR)
    return [
        min(raw_dur_s, win_sec),
        audio_rms(win),
        estimate_snr_db(win, TARGET_SR),
        spectral_rolloff_hz(win, TARGET_SR),
    ]


def _clean_eval_window(wav: np.ndarray, sr: int, row: Dict) -> np.ndarray:
    return _clean_model_window(wav, sr)


def _extract_window_feature_rows(
    rows: List[Dict],
    window_fn: Callable[[np.ndarray, int, Dict], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Load each row once, build its model-input window via ``window_fn``, return (features, labels)."""
    feats: List[List[float]] = []
    labels: List[int] = []
    for r in rows:
        wav, sr = load_wav_mono(r["path"], target_sr=TARGET_SR)
        raw_dur = wav.shape[0] / float(sr)
        win = window_fn(wav, sr, r)
        feats.append(_window_shortcut_features(raw_dur, win))
        labels.append(int(r["label"]))
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def run_shortcut_audit(manifest_path: Path, max_per_split: int = 1000) -> Dict:
    """Audit dataset for trivial acoustic/channel shortcuts before trusting neural metrics.

    Fits an L2 logistic classifier using ONLY [duration_s, rms, est_snr_db, rolloff_85_hz] on:
      1. Raw WAV files on disk.
      2. Model-input 2.5s normalized windows (what the neural network actually sees).
      3. Clean bonafide vs unseen Kokoro test windows (if both exist in test).
    Flags shortcut_warning=True if test AUROC >= 0.60 or EER <= 40%.
    """
    rows = _resolve_manifest_rows(manifest_path)
    assert_disjoint_text_splits(rows)
    train_rows, test_rows = _select_audit_rows(rows, max_per_split)

    def extract(subset: List[Dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_feats, win_feats, labels = [], [], []
        for r in subset:
            wav, sr = load_wav_mono(r["path"], target_sr=TARGET_SR)
            raw_dur = wav.shape[0] / float(sr)
            raw_feats.append([
                raw_dur,
                audio_rms(wav),
                estimate_snr_db(wav, sr),
                spectral_rolloff_hz(wav, sr),
            ])

            win = _clean_model_window(wav, sr)
            win_feats.append(_window_shortcut_features(raw_dur, win))
            labels.append(int(r["label"]))
        return (
            np.asarray(raw_feats, dtype=np.float64),
            np.asarray(win_feats, dtype=np.float64),
            np.asarray(labels, dtype=np.float64),
        )

    tr_raw, tr_win, y_tr = extract(train_rows)
    te_raw, te_win, y_te = extract(test_rows)

    w_raw, mu_r, sig_r, const_r = _fit_logistic_weights(tr_raw, y_tr)
    raw_metrics = _score_logistic(w_raw, mu_r, sig_r, const_r, te_raw, y_te)

    w_win, mu_w, sig_w, const_w = _fit_logistic_weights(tr_win, y_tr)
    win_metrics = _score_logistic(w_win, mu_w, sig_w, const_w, te_win, y_te)

    audit: Dict = {
        "disjoint_text_check": "PASSED",
        "features": SHORTCUT_FEATURE_NAMES,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "raw_acoustic_shortcuts": {
            **raw_metrics,
            "shortcut_warning": _is_shortcut(raw_metrics),
            "test_feature_means": _feature_means(te_raw, y_te),
        },
        "model_window_shortcuts": {
            **win_metrics,
            "shortcut_warning": _is_shortcut(win_metrics),
            "test_feature_means": _feature_means(te_win, y_te),
        },
    }

    clean_real_test = [r for r in test_rows if r.get("generator") == "bonafide_clean"]
    kokoro_test = [r for r in test_rows if "kokoro" in str(r.get("generator", ""))]
    if clean_real_test and kokoro_test:
        pair = clean_real_test[: min(len(clean_real_test), len(kokoro_test))] + kokoro_test[: min(len(clean_real_test), len(kokoro_test))]
        _, ck_win, y_ck = extract(pair)
        ck_metrics = _score_logistic(w_win, mu_w, sig_w, const_w, ck_win, y_ck)
        audit["clean_real_vs_kokoro_shortcuts"] = {
            **ck_metrics,
            "shortcut_warning": _is_shortcut(ck_metrics),
            "test_feature_means": _feature_means(ck_win, y_ck),
            "num_scored": len(pair),
        }

    return audit


def run_augmented_shortcut_audit(
    manifest_path: Path,
    max_per_split: int = 1000,
    seed: int = 42,
    opus_cache_dir: str | Path | None = None,
) -> Dict:
    """Shortcut audit under the same augmentation pipeline used in training.

    Builds model-input windows exactly like ``SpoofAudioDataset`` (random crop +
    cached Opus variant + ``TrainCodecAugmenter``) for train rows (augmenter
    ``seed``) and test rows (``seed + 1``), then fits the trivial 4-feature
    logistic on augmented train windows and scores augmented test windows.
    ``model_window_shortcuts`` repeats the unaugmented audit on the same rows
    for before/after comparison. The augmenter never sees labels.
    """
    rows = _resolve_manifest_rows(manifest_path)
    assert_disjoint_text_splits(rows)
    train_rows, test_rows = _select_audit_rows(rows, max_per_split)

    opus_cache_map: Dict[str, str] = {}
    cache_dir: Path | None = None
    if has_ffmpeg():
        cache_dir = Path(opus_cache_dir) if opus_cache_dir is not None else manifest_path.parent / "opus_cache_16k"
        opus_cache_map = precompute_opus_cache(
            [r["path"] for r in train_rows] + [r["path"] for r in test_rows],
            cache_dir=cache_dir,
            bitrate_kbps=16,
        )

    def make_augmented_window_fn(augmenter: TrainCodecAugmenter, crop_seed: int) -> Callable:
        crop_rng = np.random.default_rng(crop_seed)

        def window_fn(wav: np.ndarray, sr: int, row: Dict) -> np.ndarray:
            win = pad_or_crop(wav, DEFAULT_WINDOW_SAMPLES, random_crop=True, rng=crop_rng)
            cached = None
            src = row["path"]
            if src in opus_cache_map:
                opus_wav, _ = load_wav_mono(opus_cache_map[src], target_sr=TARGET_SR)
                cached = pad_or_crop(opus_wav, DEFAULT_WINDOW_SAMPLES, random_crop=True, rng=crop_rng)
            return augmenter(win, sr=sr, cached_opus_waveform=cached)

        return window_fn

    train_aug = TrainCodecAugmenter(allow_inline_opus=False, seed=seed)
    test_aug = TrainCodecAugmenter(allow_inline_opus=False, seed=seed + 1)

    tr_clean, y_tr = _extract_window_feature_rows(train_rows, _clean_eval_window)
    te_clean, y_te = _extract_window_feature_rows(test_rows, _clean_eval_window)
    tr_aug, _ = _extract_window_feature_rows(train_rows, make_augmented_window_fn(train_aug, seed))
    te_aug, _ = _extract_window_feature_rows(test_rows, make_augmented_window_fn(test_aug, seed + 1))

    w_clean, mu_c, sig_c, const_c = _fit_logistic_weights(tr_clean, y_tr)
    clean_metrics = _score_logistic(w_clean, mu_c, sig_c, const_c, te_clean, y_te)

    w_aug, mu_a, sig_a, const_a = _fit_logistic_weights(tr_aug, y_tr)
    aug_metrics = _score_logistic(w_aug, mu_a, sig_a, const_a, te_aug, y_te)

    return {
        "disjoint_text_check": "PASSED",
        "features": SHORTCUT_FEATURE_NAMES,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "seed": seed,
        "model_window_shortcuts": {
            **clean_metrics,
            "shortcut_warning": _is_shortcut(clean_metrics),
            "test_feature_means": _feature_means(te_clean, y_te),
        },
        "augmented_train_window_shortcuts": {
            **aug_metrics,
            "shortcut_warning": _is_shortcut(aug_metrics),
            "test_feature_means": _feature_means(te_aug, y_te),
        },
        "augmentation": {
            "allow_inline_opus": False,
            "p_codec": train_aug.p_codec,
            "p_noise": train_aug.p_noise,
            "p_clip": train_aug.p_clip,
            "p_lowpass": train_aug.p_lowpass,
            "p_low_shelf": train_aug.p_low_shelf,
            "p_high_shelf": train_aug.p_high_shelf,
            "p_tilt": train_aug.p_tilt,
            "spectral_param_ranges": SPECTRAL_AUG_RANGES,
            "opus_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "opus_cached_clips": len(opus_cache_map),
        },
    }


def evaluate_slice(
    rows: List[Dict],
    detector: DetectorRuntime,
    condition: EvalCondition = "clean",
    baseline: BaselineDetectorRuntime | None = None,
) -> Dict:
    labels: List[int] = []
    cb_scores: List[float] = []
    base_scores: List[float] = []
    latencies: List[float] = []

    for i, row in enumerate(rows):
        wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
        degraded = degrade_for_eval(wav, condition=condition, sr=sr, seed=1000 + i)
        res = detector.predict(degraded)
        labels.append(int(row["label"]))
        cb_scores.append(res.spoof_prob)
        latencies.append(res.latency_ms)
        if baseline is not None:
            base_scores.append(baseline.predict_prob(degraded))

    out: Dict = {
        "condition": condition,
        "num_clips": len(labels),
        "num_real": int(sum(1 for x in labels if x == 0)),
        "num_spoof": int(sum(1 for x in labels if x == 1)),
        "cloakbreak": compute_eer_and_auroc(labels, cb_scores),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }
    if baseline is not None:
        out["baseline"] = compute_eer_and_auroc(labels, base_scores)
    return out


def evaluate_clean_real_fpr(
    clean_real_rows: List[Dict],
    detector: DetectorRuntime,
    baseline: BaselineDetectorRuntime | None = None,
    condition: EvalCondition = "clean",
) -> Dict:
    """Measure false-positive rate on clean human speech alone (predicts live laptop-mic FP risk)."""
    cb_scores: List[float] = []
    base_scores: List[float] = []
    for i, row in enumerate(clean_real_rows):
        wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
        degraded = degrade_for_eval(wav, condition=condition, sr=sr, seed=2000 + i)
        cb_scores.append(detector.predict(degraded).spoof_prob)
        if baseline is not None:
            base_scores.append(baseline.predict_prob(degraded))

    arr = np.asarray(cb_scores, dtype=np.float64)
    out: Dict = {
        "condition": condition,
        "num_clean_real": len(cb_scores),
        "cloakbreak_fpr_at_05_pct": round(float(np.mean(arr >= 0.5) * 100.0), 2) if arr.size else float("nan"),
        "cloakbreak_mean_prob": round(float(arr.mean()), 4) if arr.size else float("nan"),
        "cloakbreak_p95_prob": round(float(np.percentile(arr, 95)), 4) if arr.size else float("nan"),
    }
    if baseline is not None and base_scores:
        barr = np.asarray(base_scores, dtype=np.float64)
        out["baseline_fpr_at_05_pct"] = round(float(np.mean(barr >= 0.5) * 100.0), 2)
        out["baseline_mean_prob"] = round(float(barr.mean()), 4)
    return out


def build_rolloff_matched_pair(
    reals: List[Dict],
    spoofs: List[Dict],
    n_bins: int = 10,
    max_total: int | None = None,
    seed: int = 42,
) -> tuple[List[Dict], Dict]:
    """Balance real/spoof pools across spectral-rolloff bins.

    Computes rolloff_85 on the clean model-input window for each candidate,
    bins the pooled rolloffs into quantile bins, and samples an equal number of
    real and spoof rows per bin so the trivial rolloff shortcut cannot separate
    the classes. Candidate pools are capped at 4*max_total via seeded shuffle to
    bound CPU time; when the matched total exceeds ``max_total`` each bin's k is
    scaled down proportionally (floor) keeping real==spoof per bin.
    Returns (matched_rows, summary).
    """
    rng = np.random.default_rng(seed)
    pool_cap = 4 * max_total if max_total is not None else None

    def capped(pool: List[Dict]) -> List[Dict]:
        if pool_cap is None or len(pool) <= pool_cap:
            return list(pool)
        idx = rng.permutation(len(pool))[:pool_cap]
        return [pool[int(i)] for i in idx]

    real_pool = capped(reals)
    spoof_pool = capped(spoofs)

    def pool_rolloffs(pool: List[Dict]) -> np.ndarray:
        out = np.zeros(len(pool), dtype=np.float64)
        for i, row in enumerate(pool):
            wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
            win = _clean_model_window(wav, sr)
            out[i] = spectral_rolloff_hz(win, TARGET_SR)
        return out

    real_r = pool_rolloffs(real_pool)
    spoof_r = pool_rolloffs(spoof_pool)

    before = {
        "real": round(float(real_r.mean()), 2) if real_r.size else None,
        "spoof": round(float(spoof_r.mean()), 2) if spoof_r.size else None,
    }

    summary: Dict = {
        "n_bins": n_bins,
        "max_total": max_total,
        "pool_cap": pool_cap,
        "seed": seed,
        "pool_sizes": {"real": len(real_pool), "spoof": len(spoof_pool)},
        "n_bins_used": 0,
        "per_bin": [],
        "rolloff_mean_hz": {"before": before, "after": {"real": None, "spoof": None}},
    }
    if not real_pool or not spoof_pool:
        return [], summary

    pooled = np.concatenate([real_r, spoof_r])
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([float(pooled.min()) - 1e-6, float(pooled.max()) + 1e-6])

    def bin_index(rolloffs: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges, rolloffs, side="right") - 1, 0, edges.size - 2)

    real_bins = bin_index(real_r)
    spoof_bins = bin_index(spoof_r)
    n_bins_eff = edges.size - 1

    k_avail = np.zeros(n_bins_eff, dtype=np.int64)
    for b in range(n_bins_eff):
        k_avail[b] = min(int(np.sum(real_bins == b)), int(np.sum(spoof_bins == b)))

    total_items = int(2 * k_avail.sum())
    if max_total is not None and total_items > max_total and total_items > 0:
        k_final = np.floor(k_avail * (max_total / float(total_items))).astype(np.int64)
    else:
        k_final = k_avail

    pair: List[Dict] = []
    sel_real_idx: List[int] = []
    sel_spoof_idx: List[int] = []
    per_bin: List[Dict] = []
    for b in range(n_bins_eff):
        k = int(k_final[b])
        if k <= 0:
            continue
        ri = np.flatnonzero(real_bins == b)
        si = np.flatnonzero(spoof_bins == b)
        take_r = rng.choice(ri, size=min(k, ri.size), replace=False)
        take_s = rng.choice(si, size=min(k, si.size), replace=False)
        sel_real_idx.extend(int(i) for i in take_r)
        sel_spoof_idx.extend(int(i) for i in take_s)
        for i in take_r:
            pair.append(real_pool[int(i)])
        for i in take_s:
            pair.append(spoof_pool[int(i)])
        per_bin.append(
            {
                "lo_hz": round(float(edges[b]), 2),
                "hi_hz": round(float(edges[b + 1]), 2),
                "n_real": int(take_r.size),
                "n_spoof": int(take_s.size),
            }
        )

    if sel_real_idx:
        after_real = float(real_r[np.asarray(sel_real_idx, dtype=np.int64)].mean())
    else:
        after_real = float("nan")
    if sel_spoof_idx:
        after_spoof = float(spoof_r[np.asarray(sel_spoof_idx, dtype=np.int64)].mean())
    else:
        after_spoof = float("nan")

    summary["n_bins_used"] = len(per_bin)
    summary["per_bin"] = per_bin
    summary["rolloff_mean_hz"]["after"] = {
        "real": round(after_real, 2) if np.isfinite(after_real) else None,
        "spoof": round(after_spoof, 2) if np.isfinite(after_spoof) else None,
    }
    return pair, summary


def check_int8_parity(
    manifest_path: Path,
    fp32_runtime: DetectorRuntime,
    int8_runtime: DetectorRuntime,
    max_auroc_diff: float = 0.01,
    max_clips: int | None = None,
) -> Dict:
    """Score the manifest ``val`` split (non-Kokoro) with FP32 vs INT8 ONNX runtimes.

    Returns per-runtime EER/AUROC plus probability deltas; ``passed`` is True when
    ``|auroc_fp32 - auroc_int8| <= max_auroc_diff``.
    """
    rows = _resolve_manifest_rows(manifest_path)
    val_rows = [
        r for r in rows
        if r.get("split") == "val" and "kokoro" not in str(r.get("generator", ""))
    ]
    if not val_rows:
        raise ValueError(
            f"check_int8_parity requires manifest rows with split=='val' (non-Kokoro); "
            f"{manifest_path} has none."
        )
    if max_clips is not None and len(val_rows) > max_clips:
        keep = np.random.default_rng(42).permutation(len(val_rows))[: max(1, int(max_clips))]
        val_rows = [val_rows[int(i)] for i in sorted(keep)]

    labels: List[int] = []
    fp32_scores: List[float] = []
    int8_scores: List[float] = []
    for i, row in enumerate(val_rows):
        wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
        degraded = degrade_for_eval(wav, condition="clean", sr=sr, seed=3000 + i)
        labels.append(int(row["label"]))
        fp32_scores.append(fp32_runtime.predict(degraded).spoof_prob)
        int8_scores.append(int8_runtime.predict(degraded).spoof_prob)

    m32 = compute_eer_and_auroc(labels, fp32_scores)
    m8 = compute_eer_and_auroc(labels, int8_scores)
    prob_diffs = np.abs(np.asarray(fp32_scores, dtype=np.float64) - np.asarray(int8_scores, dtype=np.float64))
    auroc_diff = abs(float(m32["auroc"]) - float(m8["auroc"]))
    return {
        "num_clips": len(val_rows),
        "num_real": int(sum(1 for v in labels if v == 0)),
        "num_spoof": int(sum(1 for v in labels if v == 1)),
        "fp32": {"auroc": m32["auroc"], "eer_pct": m32["eer_pct"]},
        "int8": {"auroc": m8["auroc"], "eer_pct": m8["eer_pct"]},
        "auroc_abs_diff": round(auroc_diff, 6),
        "mean_abs_prob_diff": round(float(prob_diffs.mean()), 6) if prob_diffs.size else 0.0,
        "max_abs_prob_diff": round(float(prob_diffs.max()), 6) if prob_diffs.size else 0.0,
        "max_auroc_diff": float(max_auroc_diff),
        "passed": bool(auroc_diff <= float(max_auroc_diff)),
    }


def save_audit_json(audit: Dict, results_dir: Path, kind: str, manifest_path: Path) -> Path:
    """Persist an audit JSON under ``results/{kind}_{manifest_parent}_{UTC ts}.json``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"{kind}_{manifest_path.parent.name}_{stamp}.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[audit-json] wrote {out_path}")
    return out_path


def format_markdown_table(results: Dict[str, Dict]) -> str:
    lines = [
        "| Eval Slice | Clips (Real/Spoof) | Baseline EER (%) | CloakBreak EER (%) | CloakBreak AUROC | p50 / p95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, r in results.items():
        if "num_clips" not in r:
            continue
        base_eer = f"{r['baseline']['eer_pct']:.2f}" if "baseline" in r else "N/A"
        cb_eer = f"{r['cloakbreak']['eer_pct']:.2f}"
        cb_auc = f"{r['cloakbreak']['auroc']:.4f}"
        lat = f"{r['latency_p50_ms']:.1f} / {r['latency_p95_ms']:.1f}"
        counts = f"{r['num_clips']} ({r['num_real']}/{r['num_spoof']})"
        lines.append(f"| {name} | {counts} | {base_eer} | {cb_eer} | {cb_auc} | {lat} |")

    audit = results.get("shortcut_audit") or {}
    audit_num_test = audit.get("num_test")
    model_win = audit.get("model_window_shortcuts") or {}
    if model_win:
        lines.append(
            f"| Floor: trivial 4-feature logistic (dur/RMS/SNR/rolloff) "
            f"| {audit_num_test} | N/A | {model_win['eer_pct']:.2f} | {model_win['auroc']:.4f} | N/A |"
        )
    kokoro_short = audit.get("clean_real_vs_kokoro_shortcuts") or {}
    if kokoro_short:
        lines.append(
            f"| Floor: trivial 4-feature logistic (clean real vs Kokoro) "
            f"| {kokoro_short.get('num_scored', audit_num_test)} | N/A | {kokoro_short['eer_pct']:.2f} | "
            f"{kokoro_short['auroc']:.4f} | N/A |"
        )

    if "clean_real_fpr" in results:
        fpr = results["clean_real_fpr"]
        base_str = f" | Baseline FPR@0.5: {fpr['baseline_fpr_at_05_pct']:.2f}%" if "baseline_fpr_at_05_pct" in fpr else ""
        lines.append("")
        lines.append(
            f"Clean human speech FPR@0.5 ({fpr['num_clean_real']} clips): "
            f"CloakBreak = {fpr['cloakbreak_fpr_at_05_pct']:.2f}% "
            f"(mean prob {fpr['cloakbreak_mean_prob']:.4f}, p95 {fpr['cloakbreak_p95_prob']:.4f}){base_str}"
        )
    return "\n".join(lines)


def run_evaluation(
    manifest_path: Path,
    detector: DetectorRuntime,
    baseline: BaselineDetectorRuntime | None = None,
    max_per_slice: int = 500,
    out_json: Path | None = None,
    results_dir: Path | None = None,
) -> Dict[str, Dict]:
    rows = _resolve_manifest_rows(manifest_path)
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        test_rows = rows

    real_test = [r for r in test_rows if int(r["label"]) == 0]
    clean_real_test = [r for r in real_test if r.get("generator") == "bonafide_clean"]
    main_spoof = [r for r in test_rows if int(r["label"]) == 1 and "kokoro" not in str(r.get("generator", ""))]
    kokoro_spoof = [r for r in test_rows if int(r["label"]) == 1 and "kokoro" in str(r.get("generator", ""))]

    def balanced_pair(reals: List[Dict], spoofs: List[Dict]) -> List[Dict]:
        n = min(len(reals), len(spoofs), max_per_slice // 2)
        if n == 0:
            return (reals + spoofs)[:max_per_slice]
        return reals[:n] + spoofs[:n]

    main_pair = balanced_pair(real_test, main_spoof)
    kokoro_all_pair = balanced_pair(real_test, kokoro_spoof) if kokoro_spoof else []
    clean_vs_kokoro_pair = balanced_pair(clean_real_test, kokoro_spoof) if (clean_real_test and kokoro_spoof) else []

    slices: Dict[str, Dict] = {
        "shortcut_audit": run_shortcut_audit(manifest_path, max_per_split=max_per_slice),
    }
    for cond in ("clean", "opus_16k", "phone_8k", "noise_10db", "noise_5db", "short_0.5s", "short_0.8s"):
        slices[f"test_{cond}"] = evaluate_slice(main_pair, detector, condition=cond, baseline=baseline)
    if kokoro_all_pair:
        slices["unseen_kokoro_clean"] = evaluate_slice(kokoro_all_pair, detector, condition="clean", baseline=baseline)
        slices["unseen_kokoro_opus_16k"] = evaluate_slice(kokoro_all_pair, detector, condition="opus_16k", baseline=baseline)
    if clean_vs_kokoro_pair:
        slices["clean_real_vs_kokoro_clean"] = evaluate_slice(
            clean_vs_kokoro_pair, detector, condition="clean", baseline=baseline
        )
        slices["clean_real_vs_kokoro_opus_16k"] = evaluate_slice(
            clean_vs_kokoro_pair, detector, condition="opus_16k", baseline=baseline
        )

    if real_test and main_spoof:
        matched, rolloff_summary = build_rolloff_matched_pair(
            real_test, main_spoof, max_total=max_per_slice, seed=42
        )
        if matched:
            res = evaluate_slice(matched, detector, condition="clean", baseline=baseline)
            res["rolloff_matching"] = rolloff_summary
            slices["rolloff_matched_clean"] = res
    if real_test and kokoro_spoof:
        matched_k, rolloff_summary_k = build_rolloff_matched_pair(
            real_test, kokoro_spoof, max_total=max_per_slice, seed=42
        )
        if matched_k:
            res_k = evaluate_slice(matched_k, detector, condition="clean", baseline=baseline)
            res_k["rolloff_matching"] = rolloff_summary_k
            slices["rolloff_matched_unseen_kokoro_clean"] = res_k

    if clean_real_test:
        slices["clean_real_fpr"] = evaluate_clean_real_fpr(
            clean_real_test[:max_per_slice], detector, baseline=baseline, condition="clean"
        )

    if results_dir is not None:
        save_audit_json(slices["shortcut_audit"], Path(results_dir), "eval_shortcut_audit", manifest_path)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(slices, indent=2), encoding="utf-8")
    return slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CloakBreak vs Legacy Baseline")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["mock", "onnx", "torch"], default="mock")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--baseline-id", type=str, default=None, help=f"Optional HF baseline e.g. {DEFAULT_BASELINE_ID}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-per-slice", type=int, default=600)
    parser.add_argument("--out-json", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--out-markdown", type=Path, default=None,
                        help="Also write the markdown results table (with backend/model/baseline header) here")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--int8-parity", action="store_true",
                        help="Score val split with --fp32-model vs --model-path ONNX and check AUROC parity")
    parser.add_argument("--fp32-model", type=Path, default=None, help="FP32 ONNX path for --int8-parity")
    parser.add_argument("--max-auroc-diff", type=float, default=0.01, help="INT8 parity AUROC tolerance")
    parser.add_argument("--parity-max-clips", type=int, default=None, help="Cap val clips for --int8-parity")
    parser.add_argument("--onnx-threads", type=int, default=min(os.cpu_count() or 2, 8),
                        help="ONNX intra-op threads for --int8-parity")
    parser.add_argument("--shortcut-only", action="store_true", help="Run only acoustic shortcut audit & disjoint check")
    parser.add_argument(
        "--augment-train-windows",
        action="store_true",
        help="With --shortcut-only: audit trivial features on augmentation-pipeline windows (train seed, test seed+1)",
    )
    parser.add_argument(
        "--fail-on-shortcut",
        action="store_true",
        help="Exit non-zero when shortcut_warning is true unless ALLOW_SHORTCUT_WARNING=1",
    )
    args = parser.parse_args()

    allow_shortcut = os.getenv("ALLOW_SHORTCUT_WARNING", "0").strip().lower() in {"1", "true", "yes"}

    def _has_shortcut_warning(audit: Dict) -> bool:
        if audit.get("model_window_shortcuts", {}).get("shortcut_warning"):
            return True
        if audit.get("clean_real_vs_kokoro_shortcuts", {}).get("shortcut_warning"):
            return True
        return False

    if args.shortcut_only:
        if args.augment_train_windows:
            audit = run_augmented_shortcut_audit(args.manifest, max_per_split=args.max_per_slice)
            print(json.dumps(audit, indent=2))
            aug_metrics = audit["augmented_train_window_shortcuts"]
            unaug_auc = audit["model_window_shortcuts"]["auroc"]
            print(
                f"[augmented audit] trivial AUROC unaugmented={unaug_auc:.4f} "
                f"augmented={aug_metrics['auroc']:.4f} (threshold 0.60)"
            )
            save_audit_json(audit, args.results_dir, "shortcut_audit_augmented", args.manifest)
            if aug_metrics.get("shortcut_warning"):
                print("[SHORTCUT WARNING] Trivial acoustic classifier still exceeds chance on augmented model windows!")
                if args.fail_on_shortcut and not allow_shortcut:
                    sys.exit(2)
            return

        audit = run_shortcut_audit(args.manifest, max_per_split=args.max_per_slice)
        print(json.dumps(audit, indent=2))
        save_audit_json(audit, args.results_dir, "shortcut_audit", args.manifest)
        if _has_shortcut_warning(audit):
            print("[SHORTCUT WARNING] Trivial acoustic classifier exceeded chance on model input windows!")
            if args.fail_on_shortcut and not allow_shortcut:
                sys.exit(2)
        return

    if args.int8_parity:
        if args.fp32_model is None:
            raise SystemExit("--fp32-model is required for --int8-parity")
        if args.model_path is None:
            raise SystemExit("--model-path (INT8 ONNX) is required for --int8-parity")
        fp32_rt = OnnxDetectorRuntime(args.fp32_model, num_threads=args.onnx_threads)
        int8_rt = OnnxDetectorRuntime(args.model_path, num_threads=args.onnx_threads)
        parity = check_int8_parity(
            args.manifest,
            fp32_rt,
            int8_rt,
            max_auroc_diff=args.max_auroc_diff,
            max_clips=args.parity_max_clips,
        )
        verdict = "PASS" if parity["passed"] else "FAIL"
        print(
            f"[int8 parity] fp32 AUROC={parity['fp32']['auroc']:.4f} "
            f"int8 AUROC={parity['int8']['auroc']:.4f} "
            f"|diff|={parity['auroc_abs_diff']:.4f} (max {parity['max_auroc_diff']:.4f}) {verdict}",
            flush=True,
        )
        save_audit_json(parity, args.results_dir, "int8_parity", args.manifest)
        if not parity["passed"]:
            sys.exit(4)
        return

    if args.backend == "mock":
        detector: DetectorRuntime = MockDetectorRuntime()
    elif args.backend == "onnx":
        if args.model_path is None:
            raise SystemExit("--model-path required for onnx backend")
        detector = OnnxDetectorRuntime(args.model_path)
    else:
        if args.model_path is None:
            raise SystemExit("--model-path required for torch backend")
        detector = TorchDetectorRuntime(args.model_path, device=args.device)

    baseline = BaselineDetectorRuntime(args.baseline_id, device=args.device) if args.baseline_id else None
    results = run_evaluation(
        manifest_path=args.manifest,
        detector=detector,
        baseline=baseline,
        max_per_slice=args.max_per_slice,
        out_json=args.out_json,
        results_dir=args.results_dir,
    )
    print("=== Shortcut Audit (Duration + RMS + SNR + Spectral Rolloff) ===")
    print(json.dumps(results["shortcut_audit"], indent=2))
    print("\n=== Benchmark Table ===")
    md_table = format_markdown_table(results)
    print(md_table)
    if args.out_markdown is not None:
        args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
        header = [
            f"backend: {detector.backend_name}",
            f"model: {args.model_path if args.model_path is not None else 'n/a'}",
            f"baseline: {args.baseline_id if args.baseline_id else 'none'}",
            "",
        ]
        args.out_markdown.write_text("\n".join(header) + md_table + "\n", encoding="utf-8")
        print(f"[eval] wrote markdown table -> {args.out_markdown}")


if __name__ == "__main__":
    main()
