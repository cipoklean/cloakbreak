import { useEffect, useState } from "react";
import { getJson } from "../api";
import type { AnalysisResult, JobProgress, SampleInfo } from "../types";
import { errorText, formatScore, sampleDisplayName, truthLabel } from "../utils";
import { PlayIcon } from "./Icons";

interface SamplesPanelProps {
  inFlight: boolean;
  jobProgress: JobProgress | null;
  analysisError: string | null;
  sampleResults: Record<string, AnalysisResult>;
  currentSampleId?: string;
  onAnalyze: (sample: SampleInfo) => void;
}

export function SamplesPanel({
  inFlight,
  jobProgress,
  analysisError,
  sampleResults,
  currentSampleId,
  onAnalyze,
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
    <div data-testid="samples-panel">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-base font-medium text-cb-text">Try samples</h2>
          <p className="mt-1 text-sm text-cb-muted">
            Six test-split clips picked for the demo. Not a benchmark; see the results table in the README.
          </p>
        </div>
      </div>
      {analysisError && jobProgress == null ? (
        <p className="mt-4 rounded-lg border border-cb-clone bg-cb-bg px-4 py-3 text-sm font-medium text-cb-clone" role="alert" data-testid="analysis-error">
          {analysisError}
        </p>
      ) : null}
      {loadError ? (
        <p className="mt-4 rounded-lg border border-cb-clone bg-cb-bg px-4 py-3 text-sm font-medium text-cb-clone" role="alert">
          {loadError}
        </p>
      ) : null}
      {!samples && !loadError ? <p className="mt-4 text-sm text-cb-muted">Loading samples…</p> : null}
      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
        {samples?.map((sample) => {
          const result = sampleResults[sample.id];
          const label = sampleDisplayName(sample.id);
          const analyzingThis = inFlight && jobProgress?.kind === "sample" && jobProgress.sourceId === sample.id;
          const selected = currentSampleId === sample.id;
          const action = `Analyze clip ${label}`;
          return (
            <div key={sample.id} className="min-w-0" data-testid={`sample-card-${sample.id}`}>
              <button
                type="button"
                className={`flex h-full min-h-24 w-full flex-col rounded-lg border p-3 text-left transition-[background-color,border-color,transform] duration-150 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-55 disabled:active:translate-y-0 ${
                  selected
                    ? "border-cb-secondary bg-cb-bg"
                    : sample.available
                      ? "border-cb-border bg-cb-bg hover:border-cb-muted"
                      : "border-cb-border bg-cb-bg"
                }`}
                data-testid={`analyze-sample-${sample.id}`}
                disabled={!sample.available || inFlight}
                title={!sample.available ? "This sample is not restored" : disabledTitle}
                aria-label={action}
                aria-pressed={selected}
                onClick={() => onAnalyze(sample)}
              >
                <span className="flex items-start justify-between gap-2">
                  <span className="min-w-0 break-words text-sm font-medium text-cb-text">{label}</span>
                  <PlayIcon className="mt-0.5 h-4 w-4 shrink-0 text-cb-muted" />
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
                  {sample.duration_s != null ? <span className="font-mono text-[11px] tabular-nums text-cb-muted">{sample.duration_s.toFixed(1)} s</span> : null}
                  {!sample.available ? (
                    <span className="rounded-lg border border-cb-border px-2 py-0.5 text-xs font-medium text-cb-muted">not restored</span>
                  ) : null}
                  {result ? (
                    <span
                      className={`rounded-lg border px-2 py-0.5 text-xs font-medium ${sample.truth === "clone" ? "border-cb-clone text-cb-clone" : "border-cb-real text-cb-real"}`}
                      data-testid={`truth-pill-${sample.id}`}
                    >
                      {truthLabel(sample.truth)}
                    </span>
                  ) : null}
                </span>
                {result ? (
                  <span className="mt-auto flex flex-wrap gap-x-2 gap-y-1 pt-3 font-mono text-[11px] tabular-nums" data-testid={`sample-result-${sample.id}`}>
                    <span className={result.verdict === "spoof" ? "text-cb-clone" : "text-cb-real"} data-testid={`sample-verdict-${sample.id}`}>
                      {result.verdict === "spoof" ? "Clone" : "Real"}
                    </span>
                    <span className="text-cb-muted">
                      max <strong className="text-cb-text" data-testid={`sample-max-${sample.id}`}>{formatScore(result.max_spoof_prob)}</strong>
                    </span>
                    <span className="text-cb-muted">
                      over <strong className="text-cb-text" data-testid={`sample-over-${sample.id}`}>{result.n_over_threshold}/{result.n_speech}</strong>
                    </span>
                  </span>
                ) : (
                  <span className="mt-auto pt-3 text-xs font-medium text-cb-muted">{analyzingThis ? "Analyzing…" : "Analyze clip"}</span>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
