import { useEffect, useState } from "react";
import { apiUrl, getJson } from "../api";
import type { AnalysisResult, JobProgress, SampleInfo } from "../types";
import { errorText, formatScore, sampleDisplayName, truthLabel } from "../utils";
import { Button } from "./Button";
import { JobStatusNote } from "./JobStatus";

interface SamplesPanelProps {
  inFlight: boolean;
  jobProgress: JobProgress | null;
  analysisError: string | null;
  sampleResults: Record<string, AnalysisResult>;
  currentSampleId?: string;
  onAnalyze: (sample: SampleInfo) => void;
  onShowResult: (sampleId: string) => void;
  playSegment: (request: { key: string; src: string; start?: number; end?: number }) => void;
  playingKey: string | null;
}

export function SamplesPanel({
  inFlight,
  jobProgress,
  analysisError,
  sampleResults,
  currentSampleId,
  onAnalyze,
  onShowResult,
  playSegment,
  playingKey,
}: SamplesPanelProps) {
  const [samples, setSamples] = useState<SampleInfo[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getJson<SampleInfo[]>("/api/samples", "Samples unavailable — restore data/cloakbreak_v1.")
      .then((body) => {
        if (!cancelled) setSamples(body);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(errorText(err, "Samples unavailable — restore data/cloakbreak_v1."));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const disabledTitle = inFlight ? "Wait for the current analysis to finish" : undefined;

  return (
    <div className="space-y-4" data-testid="samples-panel">
      <div>
        <h2 className="text-base font-semibold text-slate-950">Try samples</h2>
        <p className="mt-1 text-sm text-slate-600">
          Six test-split clips picked for the demo. Not a benchmark; see the results table in the README.
        </p>
      </div>
      <JobStatusNote progress={jobProgress?.kind === "sample" ? jobProgress : null} />
      {analysisError && jobProgress == null ? (
        <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-800" role="alert" data-testid="analysis-error">
          {analysisError}
        </p>
      ) : null}
      {loadError ? (
        <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-800" role="alert">
          {loadError}
        </p>
      ) : null}
      {!samples && !loadError ? <p className="text-sm text-slate-600">Loading samples…</p> : null}
      <div className="grid grid-cols-1 gap-3">
        {samples?.map((sample) => {
          const result = sampleResults[sample.id];
          const label = sampleDisplayName(sample.id);
          const playKey = `sample-full-${sample.id}`;
          const isPlaying = playingKey === playKey;
          const analyzingThis = inFlight && jobProgress?.kind === "sample" && jobProgress.sourceId === sample.id;
          return (
            <article
              key={sample.id}
              className={`rounded-xl border p-4 transition duration-150 ${sample.available ? "border-[#d5e2f0] bg-slate-50 hover:border-sky-300" : "border-slate-200 bg-slate-100 opacity-75"}`}
              data-testid={`sample-card-${sample.id}`}
            >
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                    <h3 className="break-words text-sm font-semibold text-slate-950">{label}</h3>
                    {sample.duration_s != null ? <span className="font-mono text-xs tabular-nums text-slate-600">{sample.duration_s.toFixed(1)} s</span> : null}
                    {!sample.available ? (
                      <span className="rounded-full bg-slate-200 px-2.5 py-1 text-xs font-semibold text-slate-600">not restored</span>
                    ) : null}
                    {result ? (
                      <span
                        className={`rounded-full px-2.5 py-1 text-xs font-semibold ${sample.truth === "clone" ? "bg-red-100 text-red-900" : "bg-green-100 text-green-900"}`}
                        data-testid={`truth-pill-${sample.id}`}
                      >
                        {truthLabel(sample.truth)}
                      </span>
                    ) : null}
                  </div>
                  {result ? (
                    <div className="mt-3 grid grid-cols-1 gap-2 text-sm text-slate-700 sm:grid-cols-3" data-testid={`sample-result-${sample.id}`}>
                      <div>
                        <span className="block text-xs text-slate-500">Overall verdict</span>
                        <strong className={result.verdict === "spoof" ? "text-red-700" : "text-green-700"} data-testid={`sample-verdict-${sample.id}`}>
                          {result.verdict === "spoof" ? "Clone" : "Real"}
                        </strong>
                      </div>
                      <div>
                        <span className="block text-xs text-slate-500">Max window score</span>
                        <strong className="font-mono tabular-nums" data-testid={`sample-max-${sample.id}`}>{formatScore(result.max_spoof_prob)}</strong>
                      </div>
                      <div>
                        <span className="block text-xs text-slate-500">Windows over threshold</span>
                        <strong className="font-mono tabular-nums" data-testid={`sample-over-${sample.id}`}>
                          {result.n_over_threshold}/{result.n_speech}
                        </strong>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-600">Ground truth appears after this clip is analyzed.</p>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                  {sample.available ? (
                    <Button
                      variant="secondary"
                      className="min-w-[5.5rem] px-3"
                      aria-label={`${isPlaying ? "Pause" : "Play"} ${label}`}
                      data-testid={`play-sample-${sample.id}`}
                      onClick={() => playSegment({ key: playKey, src: apiUrl(`/api/samples/${encodeURIComponent(sample.id)}/audio`) })}
                    >
                      {isPlaying ? "Pause" : "Play"}
                    </Button>
                  ) : null}
                  {result ? (
                    <Button
                      variant="secondary"
                      className="px-3"
                      data-testid={`show-result-${sample.id}`}
                      disabled={currentSampleId === sample.id}
                      onClick={() => onShowResult(sample.id)}
                    >
                      Show result
                    </Button>
                  ) : null}
                  <Button
                    variant="primary"
                    data-testid={`analyze-sample-${sample.id}`}
                    disabled={!sample.available || inFlight}
                    title={!sample.available ? "This sample is not restored" : disabledTitle}
                    aria-label={`Analyze clip ${label}`}
                    onClick={() => onAnalyze(sample)}
                  >
                    {analyzingThis ? "Analyzing…" : "Analyze clip"}
                  </Button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
