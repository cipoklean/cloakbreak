export const COLORS = {
  real: "#35d07f",
  clone: "#ff5a64",
  neutral: "#7d8994",
  text: "#f3f6f8",
  muted: "#b6c0ca",
  line: "#29313b",
};

export const MINUS = "\u2212";

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function scoreColor(score: number | null | undefined, threshold: number): string {
  if (score == null || !Number.isFinite(score)) return COLORS.neutral;
  return score >= threshold ? COLORS.clone : COLORS.real;
}

export function scoreColorClass(score: number | null | undefined, threshold: number): string {
  if (score == null || !Number.isFinite(score)) return "text-cb-muted";
  return score >= threshold ? "text-cb-clone" : "text-cb-real";
}

export function verdictWord(score: number | null | undefined, threshold: number): string {
  if (score == null || !Number.isFinite(score)) return "—";
  return score >= threshold ? "Clone" : "Real";
}

export function formatScore(score: number | null | undefined, digits = 3): string {
  if (score == null || !Number.isFinite(score)) return "—";
  return score.toFixed(digits);
}

export function formatThreshold(threshold: number | null | undefined): string {
  if (threshold == null || !Number.isFinite(threshold)) return "—";
  return threshold.toFixed(4).replace(/0+$/, "").replace(/\.$/, ".0");
}

export function formatMeasuredNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1");
}

export function formatDbfs(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--";
  const rounded = Math.round(value);
  if (rounded === 0) return "0";
  return rounded < 0 ? `${MINUS}${Math.abs(rounded)}` : String(rounded);
}

export function percentFromFraction(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function sampleDisplayName(id: string): string {
  if (id.startsWith("kokoro_")) return `Kokoro clip ${id.slice("kokoro_".length)}`;
  if (id.startsWith("ls_")) return `LibriSpeech ${id.slice("ls_".length)}`;
  return id;
}

export function truthLabel(truth: string | undefined): string {
  return truth === "clone" ? "Clone (Kokoro, unseen TTS)" : "Real (LibriSpeech)";
}

export function resultReason(result: {
  n_over_threshold: number;
  n_speech: number;
  max_spoof_prob: number | null;
  threshold: number;
}): string {
  if (result.n_speech <= 0 || result.max_spoof_prob == null) {
    return "No speech window was scored";
  }
  if (result.n_over_threshold === 0) {
    return `No speech window crossed ${formatThreshold(result.threshold)}`;
  }
  const noun = result.n_speech === 1 ? "window" : "windows";
  return `${result.n_over_threshold} of ${result.n_speech} speech ${noun} over the threshold (${formatScore(result.max_spoof_prob)} max)`;
}

export function peakWindowIndex(windows: { spoof_prob: number | null; is_speech: boolean }[]): number | null {
  let best: number | null = null;
  let bestScore = -Infinity;
  windows.forEach((window, index) => {
    if (!window.is_speech || window.spoof_prob == null) return;
    if (window.spoof_prob > bestScore) {
      best = index;
      bestScore = window.spoof_prob;
    }
  });
  return best;
}

export function errorText(error: unknown, fallback = "Request failed"): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return fallback;
}

export function jobProgressText(progress: {
  status: "submitting" | "queued" | "running";
  position?: number;
} | null): string | null {
  if (!progress) return null;
  if (progress.status === "queued") {
    const position = progress.position ?? 1;
    return `Someone else is being analyzed. You're #${position} in line.`;
  }
  if (progress.status === "running") return "Analysis is running.";
  return "Submitting the analysis.";
}
