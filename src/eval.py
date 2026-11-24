from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.audio_utils import TARGET_SR, load_wav_mono
from src.augment import EvalCondition, degrade_for_eval
from src.model import (
    DEFAULT_BASELINE_ID,
    BaselineDetectorRuntime,
    DetectorRuntime,
    MockDetectorRuntime,
    OnnxDetectorRuntime,
    TorchDetectorRuntime,
)


def compute_eer_and_auroc(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return {"eer_pct": float("nan"), "auroc": float("nan"), "f1_at_05": float("nan"), "eer_threshold": 0.5}

    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = tps / float(pos.size)
    fpr = fps / float(neg.size)
    fnr = 1.0 - tpr

    tpr_curve = np.concatenate([[0.0], tpr, [1.0]])
    fpr_curve = np.concatenate([[0.0], fpr, [1.0]])
    auroc = float(np.trapezoid(tpr_curve, fpr_curve))

    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    eer_thresh = float(s_sorted[idx])

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


def format_markdown_table(results: Dict[str, Dict]) -> str:
    lines = [
        "| Eval Slice | Clips (Real/Spoof) | Baseline EER (%) | CloakBreak EER (%) | CloakBreak AUROC | p50 / p95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, r in results.items():
        base_eer = f"{r['baseline']['eer_pct']:.2f}" if "baseline" in r else "N/A"
        cb_eer = f"{r['cloakbreak']['eer_pct']:.2f}"
        cb_auc = f"{r['cloakbreak']['auroc']:.4f}"
        lat = f"{r['latency_p50_ms']:.1f} / {r['latency_p95_ms']:.1f}"
        counts = f"{r['num_clips']} ({r['num_real']}/{r['num_spoof']})"
        lines.append(f"| {name} | {counts} | {base_eer} | {cb_eer} | {cb_auc} | {lat} |")
    return "\n".join(lines)


def run_evaluation(
    manifest_path: Path,
    detector: DetectorRuntime,
    baseline: BaselineDetectorRuntime | None = None,
    max_per_slice: int = 500,
    out_json: Path | None = None,
) -> Dict[str, Dict]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        test_rows = rows

    real_test = [r for r in test_rows if int(r["label"]) == 0]
    itw_spoof = [r for r in test_rows if int(r["label"]) == 1 and "kokoro" not in r.get("generator", "")]
    kokoro_spoof = [r for r in test_rows if int(r["label"]) == 1 and "kokoro" in r.get("generator", "")]

    def balanced_pair(reals: List[Dict], spoofs: List[Dict]) -> List[Dict]:
        n = min(len(reals), len(spoofs), max_per_slice // 2)
        if n == 0:
            return (reals + spoofs)[:max_per_slice]
        return reals[:n] + spoofs[:n]

    main_pair = balanced_pair(real_test, itw_spoof)
    kokoro_pair = balanced_pair(real_test, kokoro_spoof) if kokoro_spoof else []

    slices: Dict[str, Dict] = {}
    for cond in ("clean", "opus_16k", "phone_8k", "noise_5db", "short_0.5s", "short_0.8s"):
        slices[f"test_{cond}"] = evaluate_slice(main_pair, detector, condition=cond, baseline=baseline)
    if kokoro_pair:
        slices["unseen_kokoro_clean"] = evaluate_slice(kokoro_pair, detector, condition="clean", baseline=baseline)
        slices["unseen_kokoro_opus_16k"] = evaluate_slice(kokoro_pair, detector, condition="opus_16k", baseline=baseline)

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
    args = parser.parse_args()

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
    print(format_markdown_table(results))


if __name__ == "__main__":
    main()
