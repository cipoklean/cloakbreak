from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn

from src.audio_utils import TARGET_SR, normalize_loudness, pad_or_crop

DEFAULT_BACKBONE_ID = "microsoft/wavlm-base-plus"
FALLBACK_BACKBONE_ID = "facebook/wav2vec2-base"
DEFAULT_BASELINE_ID = "Vansh180/deepfake-audio-wav2vec2"
FALLBACK_BASELINE_ID = "garystafford/wav2vec2-deepfake-voice-detector"
DEFAULT_WINDOW_SEC = 2.5
DEFAULT_WINDOW_SAMPLES = int(TARGET_SR * DEFAULT_WINDOW_SEC)


@dataclass
class DetectionResult:
    spoof_prob: float
    attention_focus: List[float]
    latency_ms: float
    backend: str


class DetectorRuntime(Protocol):
    backend_name: str

    def predict(self, waveform: np.ndarray) -> DetectionResult: ...


class TinyTestBackbone(nn.Module):
    """Lightweight Conv1d stand-in for offline CPU tests (no HF download)."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=160, stride=80, padding=40),
            nn.GELU(),
            nn.Conv1d(32, hidden_size, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values.unsqueeze(1)
        h = self.conv(x)
        return h.transpose(1, 2)


class TemporalAttentionPool(nn.Module):
    """Attentive statistics pooling producing temporal attention focus weights.

    Note: attention weights indicate where the model focused across time frames,
    not ground-truth proof of which frames are spoofed unless validated on
    known spliced/partial-spoof benchmarks.
    """

    def __init__(self, hidden_size: int, attn_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        w_expanded = weights.unsqueeze(-1)
        mean = torch.sum(hidden_states * w_expanded, dim=1)
        var = torch.sum(((hidden_states - mean.unsqueeze(1)) ** 2) * w_expanded, dim=1)
        std = torch.sqrt(var.clamp(min=1e-6))
        pooled = torch.cat([mean, std], dim=-1)
        logit = self.head(pooled).squeeze(-1)
        return logit, weights


class CloakBreakDetector(nn.Module):
    def __init__(
        self,
        backbone_id: str = DEFAULT_BACKBONE_ID,
        attn_dim: int = 128,
        dropout: float = 0.2,
        use_tiny_test_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_id = "tiny-test" if use_tiny_test_backbone else backbone_id
        self.use_tiny_test_backbone = use_tiny_test_backbone
        if use_tiny_test_backbone:
            self.backbone = TinyTestBackbone(hidden_size=64)
            hidden_size = 64
        else:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(backbone_id)
            hidden_size = int(self.backbone.config.hidden_size)
        self.pool_head = TemporalAttentionPool(hidden_size=hidden_size, attn_dim=attn_dim, dropout=dropout)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, num_layers: int = 6) -> None:
        self.freeze_backbone()
        if self.use_tiny_test_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layers", None) if encoder is not None else None
        if layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        for layer in layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        layer_norm = getattr(encoder, "layer_norm", None)
        if layer_norm is not None:
            for p in layer_norm.parameters():
                p.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_tiny_test_backbone:
            hidden = self.backbone(input_values)
        else:
            mean = input_values.mean(dim=-1, keepdim=True)
            std = input_values.std(dim=-1, keepdim=True).clamp(min=1e-5)
            normed = (input_values - mean) / std
            out = self.backbone(input_values=normed)
            hidden = out.last_hidden_state
        return self.pool_head(hidden)


def downsample_focus(weights: np.ndarray, bins: int = 32) -> List[float]:
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return [0.0] * bins
    if arr.size == bins:
        out = arr
    else:
        chunks = np.array_split(arr, bins)
        out = np.array([float(c.mean()) if c.size else 0.0 for c in chunks], dtype=np.float32)
    max_val = float(out.max())
    if max_val > 1e-8:
        out = out / max_val
    return [round(float(v), 4) for v in out]


def export_to_onnx(
    model: CloakBreakDetector,
    onnx_path: str | Path,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    opset: int = 17,
) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, window_samples, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input_values"],
        output_names=["spoof_logit", "attention_weights"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "spoof_logit": {0: "batch"},
            "attention_weights": {0: "batch", 1: "frames"},
        },
        opset_version=opset,
        dynamo=False,
    )
    return onnx_path


def quantize_onnx_int8(fp32_onnx: str | Path, int8_onnx: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_onnx = Path(int8_onnx)
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
    )
    return int8_onnx


class OnnxDetectorRuntime:
    def __init__(
        self,
        onnx_path: str | Path,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        num_threads: int = 2,
        focus_bins: int = 32,
    ) -> None:
        import onnxruntime as ort

        self.backend_name = f"onnx:{Path(onnx_path).name}"
        self.window_samples = window_samples
        self.focus_bins = focus_bins
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def predict(self, waveform: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        fixed = normalize_loudness(pad_or_crop(waveform, self.window_samples, random_crop=False))
        inp = fixed[np.newaxis, :].astype(np.float32)
        logit, attn = self.session.run(None, {"input_values": inp})
        val = float(np.asarray(logit).reshape(-1)[0])
        prob = float(1.0 / (1.0 + np.exp(-np.clip(val, -30.0, 30.0))))
        focus = downsample_focus(np.asarray(attn)[0], bins=self.focus_bins)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            spoof_prob=round(prob, 4),
            attention_focus=focus,
            latency_ms=round(latency_ms, 2),
            backend=self.backend_name,
        )


class TorchDetectorRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path,
        backbone_id: str = DEFAULT_BACKBONE_ID,
        device: str = "cuda",
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        focus_bins: int = 32,
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.window_samples = window_samples
        self.focus_bins = focus_bins
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        meta_backbone = ckpt.get("backbone_id", backbone_id)
        use_tiny = meta_backbone == "tiny-test"
        self.model = CloakBreakDetector(backbone_id=meta_backbone, use_tiny_test_backbone=use_tiny)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.backend_name = f"torch:{self.device.type}"

    @torch.inference_mode()
    def predict(self, waveform: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        fixed = normalize_loudness(pad_or_crop(waveform, self.window_samples, random_crop=False))
        tensor = torch.from_numpy(fixed).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            logit, attn = self.model(tensor)
        prob = float(torch.sigmoid(logit).item())
        focus = downsample_focus(attn[0].float().cpu().numpy(), bins=self.focus_bins)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            spoof_prob=round(prob, 4),
            attention_focus=focus,
            latency_ms=round(latency_ms, 2),
            backend=self.backend_name,
        )


class BaselineDetectorRuntime:
    """Off-the-shelf Hugging Face legacy Wav2Vec2 deepfake baseline."""

    def __init__(
        self,
        model_id: str = DEFAULT_BASELINE_ID,
        device: str = "cuda",
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.window_samples = window_samples
        self.model_id = model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device).eval()
        self.fake_idx = self._resolve_fake_index()

    def _resolve_fake_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for k, v in id2label.items():
            if str(v).lower() in {"fake", "spoof", "deepfake", "synthetic"}:
                return int(k)
        return 1

    @torch.inference_mode()
    def predict_prob(self, waveform: np.ndarray) -> float:
        fixed = normalize_loudness(pad_or_crop(waveform, self.window_samples, random_crop=False))
        inputs = self.extractor(fixed, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return round(float(probs[self.fake_idx].item()), 4)


class MockDetectorRuntime:
    """Zero-download deterministic CPU mock backend for Day 0 tests & UI dev."""

    def __init__(self, focus_bins: int = 32) -> None:
        self.backend_name = "mock"
        self.focus_bins = focus_bins

    def predict(self, waveform: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return DetectionResult(0.5, [0.0] * self.focus_bins, 0.1, self.backend_name)
        chunks = np.array_split(x, self.focus_bins)
        energies = np.array([float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0 for c in chunks], dtype=np.float32)
        spec = np.abs(np.fft.rfft(x[: min(x.size, 8192)]))
        flatness = float(np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8))
        prob = float(np.clip(flatness * 1.4, 0.02, 0.98))
        focus = downsample_focus(energies, bins=self.focus_bins)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            spoof_prob=round(prob, 4),
            attention_focus=focus,
            latency_ms=round(latency_ms, 2),
            backend=self.backend_name,
        )
