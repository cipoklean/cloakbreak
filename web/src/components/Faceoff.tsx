import { useEffect, useRef, useState } from "react";
import type { JobProgress } from "../types";
import { clamp01, formatScore, scoreColor } from "../utils";
import { Button } from "./Button";
import { PauseIcon, PlayIcon } from "./Icons";
import { JobStatusNote } from "./JobStatus";

type FaceoffMode = "idle" | "analyzing" | "analysis" | "live";
type FaceoffVerdict = "clone" | "real" | null;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  return reduced;
}

function useCountUp(target: number | null, revealKey: string, reducedMotion: boolean): number | null {
  const [displayed, setDisplayed] = useState<number | null>(target);
  const keyRef = useRef<string | null>(null);

  useEffect(() => {
    const validTarget = target != null && Number.isFinite(target) ? target : null;
    const shouldAnimate = validTarget != null && keyRef.current !== revealKey && !reducedMotion;
    keyRef.current = revealKey;

    let frame = 0;
    let start: number | null = null;
    const tick = (now: number) => {
      if (validTarget == null || !shouldAnimate) {
        setDisplayed(validTarget);
        return;
      }
      if (start == null) start = now;
      const progress = Math.min(1, (now - start) / 400);
      const eased = 1 - (1 - progress) * (1 - progress);
      setDisplayed(validTarget * eased);
      if (progress < 1) frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [target, revealKey, reducedMotion]);

  return displayed;
}

interface DetectorPanelProps {
  title: string;
  score: number | null;
  threshold: number;
  verdict: FaceoffVerdict;
  unavailable?: boolean;
  caption: string;
  testId: string;
  revealKey: string;
  reducedMotion: boolean;
  placeholder: string | null;
}

function DetectorPanel({
  title,
  score,
  threshold,
  verdict,
  unavailable = false,
  caption,
  testId,
  revealKey,
  reducedMotion,
  placeholder,
}: DetectorPanelProps) {
  const animatedScore = useCountUp(score, revealKey, reducedMotion);
  const barScore = unavailable || score == null ? null : animatedScore;
  const width = unavailable || barScore == null ? 0 : clamp01(barScore) * 100;
  const color = unavailable || score == null ? "#7d8994" : scoreColor(score, threshold);
  const verdictText = unavailable ? "Not loaded" : verdict == null ? "No verdict" : verdict === "clone" ? "CLONE" : "REAL";
  const verdictClass = unavailable || verdict == null ? "text-cb-muted" : verdict === "clone" ? "text-cb-clone" : "text-cb-real";

  return (
    <article className="rounded-lg border border-cb-border bg-cb-bg p-5 sm:p-6" data-testid={testId}>
      <div className="flex min-h-[280px] flex-col">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="max-w-[24rem] text-base font-medium text-cb-text">{title}</h3>
          {placeholder == null ? (
            <span className="font-mono text-xs tabular-nums text-cb-muted">threshold {formatScore(threshold, threshold === 0.5 ? 1 : 4)}</span>
          ) : null}
        </div>

        {placeholder != null ? (
          <div className="flex flex-1 items-center justify-center py-10 text-center">
            <p className="max-w-xs text-lg font-medium text-cb-muted">{placeholder}</p>
          </div>
        ) : (
          <>
            <div className={`mt-7 text-[clamp(3.5rem,7vw,5.75rem)] font-medium leading-none tracking-tight ${verdictClass}`} data-testid={`${testId}-verdict`}>
              {verdictText}
            </div>
            <div className="mt-4 font-mono text-5xl font-medium leading-none tabular-nums text-cb-text" data-testid={`${testId}-score`}>
              {unavailable ? "N/A" : formatScore(animatedScore)}
            </div>
            <p className="mt-2 text-sm text-cb-muted">{caption}</p>
            <div className="mt-auto pt-7">
              <div className="relative h-3 overflow-hidden rounded-lg border border-cb-border bg-cb-surface" aria-hidden="true">
                <div
                  className="h-full rounded-lg transition-[width,background-color] duration-200 ease-out"
                  style={{ width: `${width}%`, backgroundColor: color }}
                />
                <div className="absolute bottom-0 top-0 border-l border-dashed border-cb-text/80" style={{ left: `${clamp01(threshold) * 100}%` }} />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[11px] tabular-nums text-cb-muted">
                <span>0</span>
                <span>threshold {formatScore(threshold, threshold === 0.5 ? 1 : 4)}</span>
                <span>1</span>
              </div>
            </div>
          </>
        )}
      </div>
    </article>
  );
}

interface FaceoffProps {
  mode: FaceoffMode;
  cloakVerdict: FaceoffVerdict;
  cloakScore: number | null;
  legacyScore: number | null;
  threshold: number;
  baselineLoaded: boolean;
  summaryLine: string | null;
  freshness?: string | null;
  liveWarning?: string | null;
  jobProgress: JobProgress | null;
  canPlay: boolean;
  playKey: string | null;
  playingKey: string | null;
  onPlay: (() => void) | null;
  revealKey: string;
}

export function Faceoff({
  mode,
  cloakVerdict,
  cloakScore,
  legacyScore,
  threshold,
  baselineLoaded,
  summaryLine,
  freshness,
  liveWarning,
  jobProgress,
  canPlay,
  playKey,
  playingKey,
  onPlay,
  revealKey,
}: FaceoffProps) {
  const reducedMotion = usePrefersReducedMotion();
  const placeholder =
    mode === "analyzing"
      ? "Analyzing"
      : mode === "live"
        ? "Waiting for the first scored window"
        : mode === "analysis"
          ? "No scored speech window"
          : "Pick a sample, drop a file, or start the mic";
  const cloakPlaceholder = cloakScore != null ? null : placeholder;
  const legacyPlaceholder = legacyScore != null ? null : !baselineLoaded && mode !== "idle" && mode !== "analyzing" ? null : placeholder;
  const legacyVerdict: FaceoffVerdict = legacyScore == null ? null : legacyScore >= 0.5 ? "clone" : "real";
  const playing = playKey != null && playingKey === playKey;

  return (
    <section className="cb-card faceoff-enter p-4 sm:p-6" aria-busy={mode === "analyzing"} data-testid="faceoff-card">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-medium text-cb-text">The face-off</h2>
        {canPlay && playKey && onPlay ? (
          <Button variant="secondary" className="px-3" data-testid="faceoff-play" aria-label={playing ? "Pause clip" : "Play clip"} onClick={onPlay}>
            {playing ? <PauseIcon /> : <PlayIcon />}
            {playing ? "Pause clip" : "Play clip"}
          </Button>
        ) : null}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" data-testid="detector-cards">
        <DetectorPanel
          title="CloakBreak"
          score={cloakScore}
          threshold={threshold}
          verdict={cloakVerdict}
          caption="window spoof score"
          testId="detector-card-cloakbreak"
          revealKey={revealKey}
          reducedMotion={reducedMotion}
          placeholder={cloakPlaceholder}
        />
        <DetectorPanel
          title="Legacy detector (trained on replay attacks)"
          score={legacyScore}
          threshold={0.5}
          verdict={legacyVerdict}
          unavailable={!baselineLoaded}
          caption="baseline spoof score"
          testId="detector-card-legacy"
          revealKey={revealKey}
          reducedMotion={reducedMotion}
          placeholder={legacyPlaceholder}
        />
      </div>

      <div className="mt-4 border-t border-cb-border pt-4">
        <JobStatusNote progress={jobProgress} />
        {summaryLine ? (
          <p className="mt-2 font-mono text-sm tabular-nums text-cb-secondary" data-testid="faceoff-summary">
            {summaryLine}
          </p>
        ) : null}
        {freshness ? (
          <p className="mt-2 font-mono text-xs tabular-nums text-cb-muted" data-testid="verdict-freshness">
            {freshness}
          </p>
        ) : null}
        {liveWarning ? (
          <p className="mt-3 border-t border-cb-border pt-3 text-sm leading-relaxed text-cb-secondary" data-testid="live-warning">
            {liveWarning}
          </p>
        ) : null}
      </div>
    </section>
  );
}
