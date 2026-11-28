import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from "react";
import { websocketUrl } from "../api";
import { StreamResampler16k } from "../audio/streamResampler16k.js";
import type { Condition, LiveServerMessage, LiveSnapshot, LiveWindow } from "../types";
import { formatDbfs } from "../utils";
import { Button } from "./Button";

declare global {
  interface Window {
    __cbLiveLog?: LiveServerMessage[];
  }
}

const MIC_LEVEL_SAMPLES = 4000; // 0.25 s of outgoing 16 kHz audio
const LIVE_WINDOW_CAP = 30;

interface LiveMicPanelProps {
  condition: Condition;
  snapshot: LiveSnapshot;
  setSnapshot: Dispatch<SetStateAction<LiveSnapshot>>;
  debugCapture: boolean;
  onLiveStart: () => void;
  onLiveScored: (windowKey: string) => void;
}

function floatToPcm16(float32: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function LiveMicPanel({ condition, snapshot, setSnapshot, debugCapture, onLiveStart, onLiveScored }: LiveMicPanelProps) {
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const micLevelBufRef = useRef<number[]>([]);
  const levelTimerRef = useRef<number | null>(null);
  const liveSeqRef = useRef(0);

  const updateInputLevel = useCallback(() => {
    const buf = micLevelBufRef.current;
    let db: number | null = null;
    if (buf.length) {
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      db = rms > 0 ? 20 * Math.log10(rms) : -120;
    }
    setSnapshot((prev) => ({ ...prev, inputLevelDbfs: db }));
  }, [setSnapshot]);

  const teardown = useCallback(
    async (notice: string, clearDiagnostic: boolean) => {
      const socket = wsRef.current;
      wsRef.current = null;
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          try {
            socket.close();
          } catch {
            // closing an already-closing socket is harmless
          }
        }
      }
      if (processorRef.current) {
        try {
          processorRef.current.disconnect();
        } catch {
          // already disconnected
        }
        processorRef.current = null;
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
      if (audioCtxRef.current) {
        const ctx = audioCtxRef.current;
        audioCtxRef.current = null;
        try {
          await ctx.close();
        } catch {
          // closing twice is harmless
        }
      }
      if (levelTimerRef.current != null) {
        window.clearInterval(levelTimerRef.current);
        levelTimerRef.current = null;
      }
      micLevelBufRef.current = [];
      setSnapshot((prev) => {
        const preserveDiagnostic = !clearDiagnostic && (prev.status === "busy" || prev.status === "error");
        return {
          ...prev,
          status: preserveDiagnostic ? prev.status : "idle",
          running: false,
          notice: preserveDiagnostic ? prev.error ?? prev.notice : notice,
          error: preserveDiagnostic ? prev.error : null,
          inputLevelDbfs: null,
        };
      });
    },
    [setSnapshot],
  );

  const handleServerMessage = useCallback(
    (data: LiveServerMessage) => {
      window.__cbLiveLog = window.__cbLiveLog ?? [];
      window.__cbLiveLog.push(data);
      if (window.__cbLiveLog.length > 500) window.__cbLiveLog.splice(0, window.__cbLiveLog.length - 500);

      if (data.type === "busy") {
        const detail = data.detail ?? "Someone else is using the mic. Try again in a moment.";
        setSnapshot((prev) => ({
          ...prev,
          status: "busy",
          running: false,
          error: detail,
          notice: detail,
        }));
        return;
      }
      if (data.type === "error") {
        const detail = data.detail ?? "Detector error";
        setSnapshot((prev) => ({
          ...prev,
          status: "error",
          error: detail,
          notice: detail,
        }));
        return;
      }
      if (data.type !== "prediction") return;

      const gate = {
        inputDbfs: numeric(data.input_dbfs),
        noiseFloorDbfs: numeric(data.noise_floor_dbfs),
        speechGateDbfs: numeric(data.speech_gate_dbfs),
      };
      const isSpeech = data.is_speech === true;
      const spoof = numeric(data.spoof_prob);
      const scored = data.warming_up !== true && isSpeech && spoof != null;
      const shouldAddWindow = data.warming_up !== true && (data.is_speech === false || scored);
      let freshness: string | null = null;
      if (scored) {
        const lagS = data.lag_ms != null ? (Number(data.lag_ms) / 1000).toFixed(1) : "?";
        const inference = Math.round(Number(data.inference_ms ?? data.latency_ms ?? 0));
        freshness = `Scored audio ends ${lagS} s ago · inference ${inference} ms`;
      }

      let item: LiveWindow | null = null;
      if (shouldAddWindow) {
        liveSeqRef.current += 1;
        item = {
          id: liveSeqRef.current,
          start: numeric(data.audio_start_sec) ?? 0,
          end: numeric(data.audio_end_sec) ?? numeric(data.audio_start_sec) ?? 0,
          isSpeech,
          spoofProb: scored ? spoof : null,
          baselineProb: scored ? numeric(data.baseline_prob) : null,
          windowVerdict: scored ? data.window_verdict ?? null : null,
          verdict: scored ? data.verdict ?? null : null,
          attentionFocus: scored && Array.isArray(data.attention_focus) ? data.attention_focus : null,
          inferenceMs: Number(data.inference_ms ?? 0),
          lagMs: Number(data.lag_ms ?? 0),
        };
      }

      setSnapshot((prev) => {
        const windows = item ? [...prev.windows, item].slice(-LIVE_WINDOW_CAP) : prev.windows;
        const latestScored = scored && item ? item : prev.latestScored;
        const warming = data.warming_up === true;
        const smoothingText = typeof data.smoothing === "string" ? data.smoothing.replace("_of_", " of ") : null;
        const scoredNotice = scored
          ? `This window is ${data.window_verdict === "spoof" ? "above" : "below"} the threshold · smoothed verdict ${data.verdict === "spoof" ? "Clone" : "Real"}${smoothingText ? ` (${smoothingText} rule)` : ""}`
          : "Waiting for the first scored window";
        const notice = warming
          ? `${data.window_sec != null ? Number(data.window_sec).toFixed(1) : "?"} s of 2.0 s buffered`
          : data.is_speech === false
            ? "No speech in the latest window"
            : scoredNotice;
        return {
          ...prev,
          status: "streaming",
          running: true,
          notice,
          error: null,
          freshness: scored ? freshness : prev.freshness,
          verdict: scored ? data.verdict ?? null : prev.verdict,
          windows,
          latestScored,
          voiceActivity: data.is_speech === true ? "speech" : data.is_speech === false ? "silence" : prev.voiceActivity,
          gate,
          droppedWindows: Number(data.dropped_windows ?? prev.droppedWindows),
          threshold: numeric(data.threshold) ?? prev.threshold,
        };
      });
      if (scored && item) onLiveScored(`live-${item.id}`);
    },
    [onLiveScored, setSnapshot],
  );

  const beginCapture = useCallback(
    async (socket: WebSocket) => {
      try {
        socket.send(JSON.stringify({ condition }));
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: 1,
          },
        });
        if (wsRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }
        mediaStreamRef.current = mediaStream;
        const track = mediaStream.getAudioTracks()[0];
        const settings = track && track.getSettings ? track.getSettings() : {};
        console.log("[mic] getUserMedia settings:", settings);

        let audioCtx: AudioContext | null = null;
        let source: MediaStreamAudioSourceNode;
        let resampler: StreamResampler16k | null = null;
        let resamplerMode = "";
        try {
          // Preferred path: let the browser resample to 16 kHz (band-limited).
          audioCtx = new AudioContext({ sampleRate: 16000 });
          source = audioCtx.createMediaStreamSource(mediaStream);
          resamplerMode = "browser";
        } catch (err) {
          // e.g. Firefox NotSupportedError when the stream rate differs.
          console.warn("[mic] 16 kHz AudioContext path failed, using FIR fallback:", err);
          if (audioCtx) {
            try {
              await audioCtx.close();
            } catch {
              // already closed
            }
          }
          audioCtx = new AudioContext();
          resampler = new StreamResampler16k(audioCtx.sampleRate);
          source = audioCtx.createMediaStreamSource(mediaStream);
          resamplerMode = resampler.passthrough ? "native" : `FIR fallback from ${audioCtx.sampleRate} Hz`;
        }
        if (!audioCtx) throw new Error("Could not create an AudioContext");
        if (wsRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
          mediaStream.getTracks().forEach((track) => track.stop());
          try {
            await audioCtx.close();
          } catch {
            // already closed
          }
          return;
        }
        audioCtxRef.current = audioCtx;
        const procFrames = resampler && !resampler.passthrough ? 4096 : 1024;
        const processor = audioCtx.createScriptProcessor(procFrames, 1, 1);
        processorRef.current = processor;
        processor.onaudioprocess = (event) => {
          const current = wsRef.current;
          if (!current || current.readyState !== WebSocket.OPEN) return;
          const mono = event.inputBuffer.getChannelData(0);
          const out = resampler ? resampler.process(mono) : mono;
          const levelBuf = micLevelBufRef.current;
          for (let i = 0; i < out.length; i++) levelBuf.push(out[i]);
          if (levelBuf.length > MIC_LEVEL_SAMPLES) {
            levelBuf.splice(0, levelBuf.length - MIC_LEVEL_SAMPLES);
          }
          current.send(floatToPcm16(out));
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);

        setSnapshot((prev) => ({
          ...prev,
          status: "streaming",
          running: true,
          notice: "Streaming — first verdict after 2.0 s of audio",
          error: null,
          micSettings: { ...settings },
          contextRate: audioCtx.sampleRate,
          resamplerMode,
        }));
        levelTimerRef.current = window.setInterval(updateInputLevel, 250);
        updateInputLevel();
      } catch (err) {
        const detail = err instanceof Error && err.message ? err.message : "Could not start the microphone";
        setSnapshot((prev) => ({
          ...prev,
          status: "error",
          running: false,
          error: detail,
          notice: detail,
        }));
        if (wsRef.current === socket) {
          void teardown("Microphone stopped", false);
        }
      }
    },
    [condition, setSnapshot, teardown, updateInputLevel],
  );

  const startMic = useCallback(() => {
    onLiveStart();
    window.__cbLiveLog = [];
    liveSeqRef.current = 0;
    micLevelBufRef.current = [];
    setSnapshot((prev) => ({
      ...prev,
      status: "connecting",
      running: false,
      notice: "Connecting the microphone stream",
      error: null,
      freshness: null,
      verdict: null,
      windows: [],
      latestScored: null,
      voiceActivity: null,
      inputLevelDbfs: null,
      droppedWindows: 0,
    }));
    const socket = new WebSocket(websocketUrl("/ws/detect"));
    wsRef.current = socket;
    socket.onmessage = (event) => {
      try {
        handleServerMessage(JSON.parse(String(event.data)) as LiveServerMessage);
      } catch {
        setSnapshot((prev) => ({ ...prev, status: "error", error: "Could not read the server message", notice: "Could not read the server message" }));
      }
    };
    socket.onopen = () => {
      void beginCapture(socket);
    };
    socket.onerror = () => {
      setSnapshot((prev) => ({ ...prev, status: "error", error: "The microphone websocket failed", notice: "The microphone websocket failed" }));
    };
    socket.onclose = () => {
      if (wsRef.current === socket) void teardown("Connection closed", false);
    };
  }, [beginCapture, handleServerMessage, onLiveStart, setSnapshot, teardown]);

  const stopMic = useCallback(() => {
    void teardown("Microphone stopped", true);
  }, [teardown]);

  useEffect(() => {
    const socket = wsRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ condition }));
    }
  }, [condition]);

  useEffect(() => {
    return () => {
      void teardown("Microphone stopped", true);
    };
  }, [teardown]);

  const active = snapshot.status === "connecting" || snapshot.status === "streaming";
  const level = snapshot.inputLevelDbfs;
  const levelWidth = level == null ? 0 : Math.max(0, Math.min(100, ((level + 70) / 70) * 100));
  const gateText = snapshot.gate.speechGateDbfs != null ? `${formatDbfs(snapshot.gate.speechGateDbfs)} dBFS` : "--";
  const floorText = snapshot.gate.noiseFloorDbfs != null ? ` (floor ${formatDbfs(snapshot.gate.noiseFloorDbfs)})` : "";

  return (
    <div data-testid="live-panel">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-base font-medium text-cb-text">Live mic</h2>
        <span className="rounded-lg border border-cb-border px-2 py-1 text-xs font-medium text-cb-muted">Experimental</span>
      </div>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-cb-muted">
        Streams your microphone and scores the newest 2.5 s window each time the previous one finishes. Windows below the speech gate are skipped.
      </p>

      <div className="mt-4 rounded-lg border border-cb-border bg-cb-bg p-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
          <Button
            variant={active ? "secondary" : "primary"}
            className="min-h-14 px-6 text-base"
            data-testid="live-mic-button"
            onClick={() => {
              if (active || wsRef.current != null) stopMic();
              else startMic();
            }}
          >
            {active ? "Stop mic" : "Start mic"}
          </Button>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-cb-text">Voice activity</span>
              <span
                className={`rounded-lg border px-2 py-1 text-xs font-medium ${snapshot.voiceActivity === "speech" ? "border-cb-real text-cb-real" : "border-cb-border text-cb-muted"}`}
                data-testid="voice-activity"
              >
                {snapshot.voiceActivity ?? "--"}
              </span>
              <span className="text-sm text-cb-muted" data-testid="live-notice">{snapshot.notice}</span>
            </div>
            <div className="mt-3">
              <div className="flex items-center justify-between gap-3 text-xs font-medium text-cb-muted">
                <span>Input level</span>
                <span className="font-mono tabular-nums" data-testid="input-level-number">{formatDbfs(level)} dBFS</span>
              </div>
              <div className="mt-1 h-3 overflow-hidden rounded-lg border border-cb-border bg-cb-surface" data-testid="input-level-meter" aria-hidden="true">
                <div className="h-full rounded-lg bg-cb-secondary transition-[width] duration-150" style={{ width: `${levelWidth}%` }} />
              </div>
              <p className="mt-1 font-mono text-xs tabular-nums text-cb-muted" data-testid="input-level-line">
                Input {formatDbfs(level)} dBFS · speech gate {gateText}{floorText}
              </p>
            </div>
            {debugCapture ? (
              <p className="mt-2 text-xs font-medium text-cb-secondary">Debug capture ON — saving received audio to results/live_capture/</p>
            ) : null}
          </div>
        </div>
        {snapshot.error ? (
          <p className="mt-4 rounded-lg border border-cb-clone px-3 py-2 text-sm font-medium text-cb-clone" role="alert" data-testid="live-error">
            {snapshot.error}
          </p>
        ) : null}
      </div>
    </div>
  );
}
