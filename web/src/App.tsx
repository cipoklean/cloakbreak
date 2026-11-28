import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiUrl, getJson, postForm, postJson } from "./api";
import { AttentionHeatmap } from "./components/AttentionHeatmap";
import { Faceoff } from "./components/Faceoff";
import { HonestLimits } from "./components/HonestLimits";
import { LiveMicPanel } from "./components/LiveMicPanel";
import { SamplesPanel } from "./components/SamplesPanel";
import { Tabs, type TabId } from "./components/Tabs";
import { Timeline, type TimelineWindowItem } from "./components/Timeline";
import { UploadLinkPanel } from "./components/UploadLinkPanel";
import { useAudioPlayer } from "./hooks";
import type {
  AnalysisResult,
  Condition,
  FactsResponse,
  HealthResponse,
  JobKind,
  JobProgress,
  JobView,
  LiveSnapshot,
  ResultContext,
  SampleInfo,
} from "./types";
import {
  errorText,
  formatDbfs,
  formatMeasuredNumber,
  formatScore,
  formatThreshold,
  peakWindowIndex,
  percentFromFraction,
  sampleDisplayName,
} from "./utils";

const initialLiveSnapshot: LiveSnapshot = {
  status: "idle",
  running: false,
  notice: "Start mic to stream audio",
  error: null,
  freshness: null,
  verdict: null,
  windows: [],
  latestScored: null,
  voiceActivity: null,
  inputLevelDbfs: null,
  gate: { inputDbfs: null, noiseFloorDbfs: null, speechGateDbfs: null },
  micSettings: null,
  contextRate: null,
  resamplerMode: null,
  droppedWindows: 0,
  threshold: null,
};

interface AnalysisErrorState {
  kind: JobKind;
  message: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function settingText(settings: Record<string, unknown> | null, key: string): string {
  if (!settings || settings[key] == null) return "n/a";
  return String(settings[key]);
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [facts, setFacts] = useState<FactsResponse | null>(null);
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("samples");
  const [condition, setCondition] = useState<Condition>("clean");
  const [currentResult, setCurrentResult] = useState<ResultContext | null>(null);
  const [displayMode, setDisplayMode] = useState<"analysis" | "live">("analysis");
  const [selectedWindowKey, setSelectedWindowKey] = useState<string | null>(null);
  const [inFlight, setInFlight] = useState(false);
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
  const [analysisError, setAnalysisError] = useState<AnalysisErrorState | null>(null);
  const [sampleResults, setSampleResults] = useState<Record<string, AnalysisResult>>({});
  const [liveSnapshot, setLiveSnapshot] = useState<LiveSnapshot>(initialLiveSnapshot);
  const inFlightRef = useRef(false);
  const audioPlayer = useAudioPlayer();

  useEffect(() => {
    let cancelled = false;
    getJson<HealthResponse>("/api/health", "backend unreachable")
      .then((body) => {
        if (cancelled) return;
        setHealth(body);
        setBackendReachable(true);
      })
      .catch(() => {
        if (!cancelled) setBackendReachable(false);
      });
    getJson<FactsResponse>("/api/facts", "facts unavailable")
      .then((body) => {
        if (!cancelled) setFacts(body);
      })
      .catch(() => {
        if (!cancelled) setFacts(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const submitAnalysis = useCallback(
    async (args: {
      kind: JobKind;
      label: string;
      sourceId?: string;
      submit: () => Promise<{ job_id: string }>;
      buildContext: (result: AnalysisResult) => ResultContext;
      onDone?: (result: AnalysisResult) => void;
    }) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      setInFlight(true);
      setDisplayMode("analysis");
      setCurrentResult(null);
      setSelectedWindowKey(null);
      setAnalysisError(null);
      setJobProgress({ jobId: null, kind: args.kind, label: args.label, sourceId: args.sourceId, status: "submitting" });
      try {
        const submitted = await args.submit();
        setJobProgress({ jobId: submitted.job_id, kind: args.kind, label: args.label, sourceId: args.sourceId, status: "queued", position: 1 });
        let view: JobView | null = null;
        for (;;) {
          view = await getJson<JobView>(`/api/jobs/${encodeURIComponent(submitted.job_id)}`, "Could not read the analysis job");
          if (view.status === "done" || view.status === "error") break;
          setJobProgress({
            jobId: submitted.job_id,
            kind: args.kind,
            label: args.label,
            sourceId: args.sourceId,
            status: view.status === "running" ? "running" : "queued",
            position: view.position,
          });
          await sleep(800);
        }
        if (view.status === "error") {
          throw new Error(view.error || "Analysis failed");
        }
        if (!view.result) throw new Error("Analysis finished without a result");
        const context = args.buildContext(view.result);
        setCurrentResult(context);
        const peak = peakWindowIndex(view.result.windows);
        setSelectedWindowKey(peak != null ? `analysis-${peak}` : null);
        setDisplayMode("analysis");
        args.onDone?.(view.result);
      } catch (err) {
        setAnalysisError({ kind: args.kind, message: errorText(err, "Analysis failed") });
      } finally {
        inFlightRef.current = false;
        setInFlight(false);
        setJobProgress(null);
      }
    },
    [],
  );

  const analyzeSample = useCallback(
    (sample: SampleInfo) => {
      void submitAnalysis({
        kind: "sample",
        label: sampleDisplayName(sample.id),
        sourceId: sample.id,
        submit: () =>
          postJson<{ job_id: string }>(
            `/api/analyze/sample/${encodeURIComponent(sample.id)}`,
            { condition },
            "Could not start the sample analysis",
          ),
        buildContext: (result) => ({
          result,
          inputType: "sample",
          sourceLabel: sampleDisplayName(sample.id),
          sampleId: sample.id,
          audioUrl: apiUrl(`/api/samples/${encodeURIComponent(sample.id)}/audio`),
        }),
        onDone: (result) => {
          setSampleResults((prev) => ({ ...prev, [sample.id]: result }));
        },
      });
    },
    [condition, submitAnalysis],
  );

  const analyzeUpload = useCallback(
    (file: File, objectUrl: string) => {
      void submitAnalysis({
        kind: "upload",
        label: file.name,
        submit: () => {
          const form = new FormData();
          form.append("file", file);
          form.append("condition", condition);
          return postForm<{ job_id: string }>("/api/analyze/upload", form, "Could not start the upload analysis");
        },
        buildContext: (result) => ({
          result,
          inputType: "upload",
          sourceLabel: file.name,
          audioUrl: objectUrl,
        }),
      });
    },
    [condition, submitAnalysis],
  );

  const analyzeLink = useCallback(
    (url: string) => {
      void submitAnalysis({
        kind: "url",
        label: url,
        submit: () => postJson<{ job_id: string }>("/api/analyze-url", { url, condition }, "Could not start the link analysis"),
        buildContext: (result) => ({
          result,
          inputType: "url",
          sourceLabel: url,
          audioUrl: null,
        }),
      });
    },
    [condition, submitAnalysis],
  );

  const threshold = liveSnapshot.threshold ?? currentResult?.result.threshold ?? health?.threshold ?? 0.5;

  const analysisItems: TimelineWindowItem[] = useMemo(() => {
    if (!currentResult) return [];
    return currentResult.result.windows.map((window, index) => ({
      key: `analysis-${index}`,
      rowIndex: index,
      start: window.start,
      end: window.end,
      isSpeech: window.is_speech,
      spoofProb: window.spoof_prob,
      baselineProb: window.baseline_prob,
      attentionFocus: window.attention_focus,
    }));
  }, [currentResult]);

  const liveItems: TimelineWindowItem[] = useMemo(
    () =>
      liveSnapshot.windows.map((window) => ({
        key: `live-${window.id}`,
        rowIndex: window.id,
        start: window.start,
        end: window.end,
        isSpeech: window.isSpeech,
        spoofProb: window.spoofProb,
        baselineProb: window.baselineProb,
        attentionFocus: window.attentionFocus,
      })),
    [liveSnapshot.windows],
  );

  const showingLive = displayMode === "live";
  const timelineItems = showingLive ? liveItems : analysisItems;

  const peakIndex = useMemo(
    () => (currentResult ? peakWindowIndex(currentResult.result.windows) : null),
    [currentResult],
  );
  const peakAnalysisWindow = peakIndex != null && currentResult ? currentResult.result.windows[peakIndex] : null;


  const selectedItem = timelineItems.find((item) => item.key === selectedWindowKey) ?? null;
  const attentionFocus = showingLive
    ? selectedItem?.attentionFocus ?? liveSnapshot.latestScored?.attentionFocus ?? null
    : selectedItem?.attentionFocus ?? null;
  const attentionScore = showingLive
    ? selectedItem?.spoofProb ?? liveSnapshot.latestScored?.spoofProb ?? null
    : selectedItem?.spoofProb ?? null;
  const attentionLabel = selectedItem ? `${selectedItem.start.toFixed(1)}–${selectedItem.end.toFixed(1)} s` : null;

  const detectorCloakScore = showingLive ? liveSnapshot.latestScored?.spoofProb ?? null : peakAnalysisWindow?.spoof_prob ?? null;
  const detectorLegacyScore = showingLive ? liveSnapshot.latestScored?.baselineProb ?? null : peakAnalysisWindow?.baseline_prob ?? null;

  const result = currentResult?.result ?? null;
  const freshness = showingLive && liveSnapshot.status === "streaming" ? liveSnapshot.freshness : null;
  const cloakVerdict = showingLive
    ? liveSnapshot.latestScored && liveSnapshot.verdict
      ? liveSnapshot.verdict === "spoof"
        ? "clone"
        : "real"
      : null
    : result
      ? result.verdict === "spoof"
        ? "clone"
        : "real"
      : null;
  const faceoffMode = inFlight ? "analyzing" : showingLive ? "live" : result ? "analysis" : "idle";
  const analysisSummary = result
    ? result.n_speech > 0 && result.max_spoof_prob != null
      ? `${result.n_over_threshold} of ${result.n_speech} speech ${result.n_speech === 1 ? "window" : "windows"} over the threshold (max ${formatScore(result.max_spoof_prob)})`
      : "No speech window was scored"
    : null;
  const faceoffSummary = showingLive ? liveSnapshot.error ?? liveSnapshot.notice : analysisSummary;
  const faceoffRevealKey = showingLive
    ? liveSnapshot.latestScored
      ? "live-scored"
      : "live-idle"
    : result
      ? `${currentResult?.inputType ?? "analysis"}:${currentResult?.sourceLabel ?? "result"}:${result.latency_ms}`
      : inFlight
        ? "analyzing"
        : "idle";
  const faceoffPlayKey = !showingLive && currentResult?.audioUrl
    ? `faceoff-${currentResult.inputType}-${currentResult.sampleId ?? currentResult.sourceLabel}`
    : null;
  const playFaceoff = useCallback(() => {
    if (!currentResult?.audioUrl || !faceoffPlayKey) return;
    void audioPlayer.playSegment({ key: faceoffPlayKey, src: currentResult.audioUrl, start: 0 });
  }, [audioPlayer, currentResult?.audioUrl, faceoffPlayKey]);

  const liveWarning = useMemo(() => {
    // The caveat belongs to the live-mic input only. In particular, it must
    // not appear to qualify a sample/upload/link verdict merely because the
    // user opened the live tab while that analysis result is still displayed.
    if (activeTab !== "live" || inFlight || (displayMode === "analysis" && currentResult != null)) return null;
    const base = "Live mic can false-alarm on voices and mics unlike the training data";
    if (facts && typeof facts.livemic_frac_all === "number" && typeof facts.livemic_frac_gated === "number") {
      return `${base} (${percentFromFraction(facts.livemic_frac_all)} of all windows, ${percentFromFraction(facts.livemic_frac_gated)} of scored windows on the builder's own headset).`;
    }
    return `${base}.`;
  }, [activeTab, currentResult, displayMode, facts, inFlight]);

  const onSelectTimelineWindow = useCallback(
    (item: TimelineWindowItem) => {
      setSelectedWindowKey(item.key);
      if (showingLive) return;
      const audioUrlValue = currentResult?.audioUrl;
      if (!audioUrlValue) return;
      void audioPlayer.playSegment({
        key: `window-${item.key}`,
        src: audioUrlValue,
        start: item.start,
        end: item.end,
      });
    },
    [audioPlayer, currentResult?.audioUrl, showingLive],
  );

  const headline = useMemo(() => {
    if (!facts || typeof facts.headline_cb_eer !== "number" || typeof facts.headline_baseline_eer !== "number") return null;
    return `Real speech vs unseen Kokoro clones over Opus 16 kbps: ${formatMeasuredNumber(facts.headline_cb_eer)}% error vs ${formatMeasuredNumber(facts.headline_baseline_eer)}% for a legacy detector.`;
  }, [facts]);

  const backendChip = backendReachable === false ? "backend unreachable" : health?.backend ? health.backend : "backend connecting";
  const micSettingsLine = useMemo(() => {
    if (!liveSnapshot.micSettings && liveSnapshot.contextRate == null && !liveSnapshot.resamplerMode) return null;
    const settings = liveSnapshot.micSettings;
    const deviceRate = settings?.sampleRate != null ? `${String(settings.sampleRate)} Hz` : "n/a";
    return (
      `Mic: echoCancellation=${settingText(settings, "echoCancellation")} · noiseSuppression=${settingText(settings, "noiseSuppression")}` +
      ` · autoGainControl=${settingText(settings, "autoGainControl")} · device rate ${deviceRate}` +
      ` · context ${liveSnapshot.contextRate ?? "n/a"} Hz · resampler: ${liveSnapshot.resamplerMode ?? "n/a"}` +
      (health?.debug_capture ? " · Debug capture ON — saving received audio to results/live_capture/" : "")
    );
  }, [health?.debug_capture, liveSnapshot.contextRate, liveSnapshot.micSettings, liveSnapshot.resamplerMode]);
  const inputFooterLine = micSettingsLine
    ? `Input ${formatDbfs(liveSnapshot.inputLevelDbfs)} dBFS · speech gate ${liveSnapshot.gate.speechGateDbfs != null ? `${formatDbfs(liveSnapshot.gate.speechGateDbfs)} dBFS` : "--"}${liveSnapshot.gate.noiseFloorDbfs != null ? ` (floor ${formatDbfs(liveSnapshot.gate.noiseFloorDbfs)})` : ""}`
    : null;

  const timelineMeta = !showingLive && result
    ? `${currentResult?.sourceLabel ?? "Analysis"} · overall verdict ${result.verdict === "spoof" ? "Clone" : "Real"} · rule: ${result.verdict_rule}`
    : showingLive && liveSnapshot.windows.length
      ? `Live stream · ${liveSnapshot.windows.length} recent windows · ${liveSnapshot.droppedWindows} stale windows dropped`
      : null;

  const sampleError = analysisError?.kind === "sample" ? analysisError.message : null;
  const uploadError = analysisError?.kind === "upload" ? analysisError.message : null;
  const linkError = analysisError?.kind === "url" ? analysisError.message : null;

  return (
    <div className="min-h-screen bg-cb-bg text-cb-text">
      <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 sm:py-6">
        <header className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <h1 className="text-2xl font-medium tracking-tight text-cb-text">CloakBreak</h1>
            <p className="mt-1 text-sm text-cb-secondary">Spots cloned voices, even after Opus call compression.</p>
            {headline ? <p className="mt-2 font-mono text-sm leading-relaxed tabular-nums text-cb-muted">{headline}</p> : null}
          </div>
          <div className="inline-flex w-fit max-w-full items-center rounded-lg border border-cb-border bg-cb-surface px-3 py-2 font-mono text-xs tabular-nums text-cb-muted" data-testid="backend-chip">
            <span className="break-all">{backendChip}</span>
          </div>
        </header>

        <main className="space-y-4">
          <section className="cb-card p-3" data-testid="input-card" aria-label="Input controls">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <Tabs activeTab={activeTab} onChange={setActiveTab} />
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center xl:justify-end">
                <label htmlFor="condition-select" className="whitespace-nowrap text-sm font-medium text-cb-secondary">
                  Simulate call channel
                </label>
                <select
                  id="condition-select"
                  value={condition}
                  data-testid="condition-select"
                  className="cb-input w-full px-3 py-2 text-sm sm:min-w-64"
                  onChange={(event) => setCondition(event.target.value as Condition)}
                >
                  <option value="clean">Clean</option>
                  <option value="opus_16k">Opus 16 kbps (WhatsApp-like)</option>
                  <option value="phone_8k">Phone 8 kHz</option>
                </select>
              </div>
            </div>
          </section>

          <Faceoff
            mode={faceoffMode}
            cloakVerdict={cloakVerdict}
            cloakScore={detectorCloakScore}
            legacyScore={detectorLegacyScore}
            threshold={result?.threshold ?? health?.threshold ?? liveSnapshot.threshold ?? 0.5}
            baselineLoaded={health?.baseline_loaded === true}
            summaryLine={faceoffSummary}
            freshness={freshness}
            liveWarning={liveWarning}
            jobProgress={inFlight ? jobProgress : null}
            canPlay={!showingLive && Boolean(currentResult?.audioUrl)}
            playKey={faceoffPlayKey}
            playingKey={audioPlayer.playingKey}
            onPlay={playFaceoff}
            revealKey={faceoffRevealKey}
          />

          <section className="cb-card p-4 sm:p-5" aria-label="Input details">
            <div role="tabpanel" id="panel-samples" aria-labelledby="tab-samples" hidden={activeTab !== "samples"}>
              <SamplesPanel
                inFlight={inFlight}
                jobProgress={jobProgress}
                analysisError={sampleError}
                sampleResults={sampleResults}
                currentSampleId={!showingLive ? currentResult?.sampleId : undefined}
                onAnalyze={analyzeSample}
              />
            </div>
            <div role="tabpanel" id="panel-upload" aria-labelledby="tab-upload" hidden={activeTab !== "upload"}>
              <UploadLinkPanel
                inFlight={inFlight}
                jobProgress={jobProgress}
                uploadError={uploadError}
                linkError={linkError}
                onAnalyzeUpload={analyzeUpload}
                onAnalyzeLink={analyzeLink}
              />
            </div>
            <div role="tabpanel" id="panel-live" aria-labelledby="tab-live" hidden={activeTab !== "live"}>
              <LiveMicPanel
                condition={condition}
                snapshot={liveSnapshot}
                setSnapshot={setLiveSnapshot}
                debugCapture={health?.debug_capture === true}
                onLiveStart={() => {
                  setDisplayMode("live");
                  setCurrentResult(null);
                  setSelectedWindowKey(null);
                  setAnalysisError(null);
                }}
                onLiveScored={(windowKey) => setSelectedWindowKey(windowKey)}
              />
            </div>
          </section>

          <Timeline
            items={timelineItems}
            threshold={threshold}
            selectedKey={selectedWindowKey}
            onSelect={onSelectTimelineWindow}
            canPlayAudio={!showingLive && Boolean(currentResult?.audioUrl)}
            audioUrl={!showingLive ? currentResult?.audioUrl ?? null : null}
            metaText={timelineMeta}
            emptyText={showingLive ? "Start mic to fill the live window timeline." : "Nothing analyzed yet. Scored windows will appear here."}
          />

          <AttentionHeatmap focus={attentionFocus} score={attentionScore} threshold={threshold} windowLabel={attentionLabel} />

          <HonestLimits facts={facts} />
        </main>

        <footer className="mt-5 space-y-1 font-mono text-xs tabular-nums text-cb-muted" data-testid="footer">
          <p>
            Backend {health?.backend ?? (backendReachable === false ? "unreachable" : "connecting")} · threshold {formatThreshold(result?.threshold ?? health?.threshold ?? liveSnapshot.threshold)} · source {health?.threshold_source ?? "unknown"}
          </p>
          {micSettingsLine ? <p className="break-words" data-testid="mic-settings-line">{micSettingsLine}</p> : null}
          {inputFooterLine ? <p data-testid="mic-input-footer">{inputFooterLine}</p> : null}
        </footer>
        <audio ref={audioPlayer.audioRef} className="hidden" aria-hidden="true" />
      </div>
    </div>
  );
}
