from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

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
from src.augment import EvalCondition, degrade_for_eval
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
    auroc = float(np.trapezoid(tpr_curve, fpr_curve))

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

    train_rows = [r for r in rows if r.get("split") == "train" and "kokoro" not in r.get("generator", "")][:max_per_split]
    test_rows = [r for r in rows if r.get("split") == "test"][:max_per_split]
    if not train_rows or not test_rows:
        mid = max(2, len(rows) // 2)
        train_rows, test_rows = rows[:mid], rows[mid:]

    win_sec = DEFAULT_WINDOW_SAMPLES / float(TARGET_SR)

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

            win = normalize_loudness(pad_or_crop(wav, DEFAULT_WINDOW_SAMPLES, random_crop=False))
            eff_dur = min(raw_dur, win_sec)
            win_feats.append([
                eff_dur,
                audio_rms(win),
                estimate_snr_db(win, TARGET_SR),
                spectral_rolloff_hz(win, TARGET_SR),
            ])
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
        }

    return audit


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


def format_markdown_table(results: Dict[str, Dict]) -> str:
    lines = [
        "| Eval Slice | Clips (Real/Spoof) | Baseline EER (%) | CloakBreak EER (%) | CloakBreak AUROC | p50 / p95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, r in results.items():
        if name in {"shortcut_audit", "clean_real_fpr"}:
            continue
        base_eer = f"{r['baseline']['eer_pct']:.2f}" if "baseline" in r else "N/A"
        cb_eer = f"{r['cloakbreak']['eer_pct']:.2f}"
        cb_auc = f"{r['cloakbreak']['auroc']:.4f}"
        lat = f"{r['latency_p50_ms']:.1f} / {r['latency_p95_ms']:.1f}"
        counts = f"{r['num_clips']} ({r['num_real']}/{r['num_spoof']})"
        lines.append(f"| {name} | {counts} | {base_eer} | {cb_eer} | {cb_auc} | {lat} |")

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
    for cond in ("clean", "opus_16k", "phone_8k", "noise_5db", "short_0.5s", "short_0.8s"):
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
    if clean_real_test:
        slices["clean_real_fpr"] = evaluate_clean_real_fpr(
            clean_real_test[:max_per_slice], detector, baseline=baseline, condition="clean"
        )

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
    parser.add_argument("--shortcut-only", action="store_true", help="Run only acoustic shortcut audit & disjoint check")
    args = parser.parse_args()

    if args.shortcut_only:
        audit = run_shortcut_audit(args.manifest, max_per_split=args.max_per_slice)
        print(json.dumps(audit, indent=2))
        if audit["model_window_shortcuts"]["shortcut_warning"]:
            print("[SHORTCUT WARNING] Trivial acoustic classifier exceeded chance on model input windows!")
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
    )
    print("=== Shortcut Audit (Duration + RMS + SNR + Spectral Rolloff) ===")
    print(json.dumps(results["shortcut_audit"], indent=2))
    print("\n=== Benchmark Table ===")
    print(format_markdown_table(results))


if __name__ == "__main__":
    main()
