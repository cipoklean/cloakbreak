export type Condition = "clean" | "opus_16k" | "phone_8k";

export interface HealthResponse {
  status?: string;
  backend?: string;
  baseline_loaded?: boolean;
  baseline_id?: string | null;
  clone_enabled?: boolean;
  threshold?: number;
  threshold_source?: string;
  debug_capture?: boolean;
  samples_available?: number;
}

export type FactsResponse = Record<string, number>;

export interface SampleInfo {
  id: string;
  truth: "clone" | "real" | string;
  source: string;
  available: boolean;
  duration_s: number | null;
}

export interface AnalysisWindow {
  start: number;
  end: number;
  dbfs: number;
  is_speech: boolean;
  spoof_prob: number | null;
  baseline_prob: number | null;
  window_verdict: "spoof" | "bonafide" | null;
  attention_focus: number[] | null;
}

export interface AnalysisResult {
  type: "sample" | "upload" | "url" | string;
  condition: string;
  duration_s: number;
  threshold: number;
  verdict: "spoof" | "bonafide";
  verdict_rule: string;
  n_windows: number;
  n_speech: number;
  n_over_threshold: number;
  frac_over_threshold: number;
  max_spoof_prob: number | null;
  latency_ms: number;
  backend: string;
  windows: AnalysisWindow[];
  sample_id?: string;
  truth?: string;
  url?: string;
}

export interface JobView {
  job_id: string;
  status: "queued" | "running" | "done" | "error" | string;
  kind: string;
  position?: number;
  result?: AnalysisResult;
  error?: string;
  error_type?: string;
}

export type JobKind = "sample" | "upload" | "url";

export interface JobProgress {
  jobId: string | null;
  kind: JobKind;
  label: string;
  sourceId?: string;
  status: "submitting" | "queued" | "running";
  position?: number;
}

export interface ResultContext {
  result: AnalysisResult;
  inputType: JobKind;
  sourceLabel: string;
  sampleId?: string;
  audioUrl?: string | null;
}

export interface LiveWindow {
  id: number;
  start: number;
  end: number;
  isSpeech: boolean;
  spoofProb: number | null;
  baselineProb: number | null;
  windowVerdict: "spoof" | "bonafide" | null;
  verdict: "spoof" | "bonafide" | null;
  attentionFocus: number[] | null;
  inferenceMs: number;
  lagMs: number;
}

export interface LiveGateInfo {
  inputDbfs: number | null;
  noiseFloorDbfs: number | null;
  speechGateDbfs: number | null;
}

export type LiveStatus = "idle" | "connecting" | "streaming" | "error" | "busy";

export interface LiveSnapshot {
  status: LiveStatus;
  running: boolean;
  notice: string;
  error: string | null;
  freshness: string | null;
  verdict: "spoof" | "bonafide" | null;
  windows: LiveWindow[];
  latestScored: LiveWindow | null;
  voiceActivity: "speech" | "silence" | null;
  inputLevelDbfs: number | null;
  gate: LiveGateInfo;
  micSettings: Record<string, unknown> | null;
  contextRate: number | null;
  resamplerMode: string | null;
  droppedWindows: number;
  threshold: number | null;
}

export interface LiveServerMessage {
  type?: string;
  detail?: string;
  warming_up?: boolean;
  is_speech?: boolean;
  spoof_prob?: number | null;
  baseline_prob?: number | null;
  window_verdict?: "spoof" | "bonafide" | null;
  verdict?: "spoof" | "bonafide" | null;
  smoothing?: string;
  recent_scores?: number[];
  threshold?: number;
  audio_start_sec?: number;
  audio_end_sec?: number;
  stream_sec_received?: number;
  inference_ms?: number;
  lag_ms?: number;
  dropped_windows?: number;
  attention_focus?: number[];
  latency_ms?: number;
  backend?: string | null;
  condition?: string;
  window_sec?: number;
  input_dbfs?: number | null;
  noise_floor_dbfs?: number | null;
  speech_gate_dbfs?: number | null;
}
