import { useEffect, useState, type KeyboardEvent } from "react";
import { useElementWidth } from "../hooks";
import { clamp01, formatScore, scoreColor, scoreColorClass } from "../utils";

export interface TimelineWindowItem {
  key: string;
  rowIndex: number;
  start: number;
  end: number;
  isSpeech: boolean;
  spoofProb: number | null;
  baselineProb: number | null;
  attentionFocus: number[] | null;
}

interface TimelineProps {
  items: TimelineWindowItem[];
  threshold: number;
  selectedKey: string | null;
  onSelect: (item: TimelineWindowItem) => void;
  canPlayAudio: boolean;
  audioUrl?: string | null;
  emptyText?: string;
  metaText?: string | null;
}



interface WaveformPeaks {
  url: string;
  values: number[];
}

function useWaveformPeaks(audioUrl: string | null | undefined, duration: number): number[] | null {
  const [decoded, setDecoded] = useState<WaveformPeaks | null>(null);

  useEffect(() => {
    if (!audioUrl || !Number.isFinite(duration) || duration <= 0) return;
    let cancelled = false;
    let context: AudioContext | null = null;

    const load = async () => {
      try {
        const response = await fetch(audioUrl);
        if (!response.ok) throw new Error(`audio fetch failed: ${response.status}`);
        const bytes = await response.arrayBuffer();
        context = new AudioContext();
        const audio = await context.decodeAudioData(bytes);
        const count = Math.max(32, Math.ceil(duration * 32));
        const values = new Array<number>(count).fill(0);
        const channels = Array.from({ length: audio.numberOfChannels }, (_, index) => audio.getChannelData(index));
        const samplesPerBucket = audio.sampleRate / 32;
        for (let bucket = 0; bucket < count; bucket += 1) {
          const start = Math.floor(bucket * samplesPerBucket);
          const end = Math.min(audio.length, Math.max(start + 1, Math.floor((bucket + 1) * samplesPerBucket)));
          if (start >= audio.length) break;
          let peak = 0;
          for (const channel of channels) {
            for (let index = start; index < end; index += 1) {
              const amplitude = Math.abs(channel[index] ?? 0);
              if (amplitude > peak) peak = amplitude;
            }
          }
          values[bucket] = Math.sqrt(clamp01(peak));
        }
        if (!cancelled) setDecoded({ url: audioUrl, values });
      } catch {
        if (!cancelled) setDecoded({ url: audioUrl, values: [] });
      } finally {
        if (context) void context.close().catch(() => undefined);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [audioUrl, duration]);

  if (!decoded || decoded.url !== audioUrl || decoded.values.length === 0) return null;
  return decoded.values;
}

function timeLabel(value: number): string {
  return `${value.toFixed(1)} s`;
}

export function Timeline({ items, threshold, selectedKey, onSelect, canPlayAudio, audioUrl, emptyText, metaText }: TimelineProps) {
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const height = 150;
  const padL = 40;
  const padR = 12;
  const padT = 16;
  const padB = 28;
  const minWindowPx = 44;
  const contentWidth = Math.max(width, padL + padR + items.length * minWindowPx);
  const plotW = Math.max(40, contentWidth - padL - padR);
  const plotH = height - padT - padB;
  const baseY = padT + plotH;
  const maxTime = Math.max(2.5, ...items.map((item) => item.end));
  const waveformPeaks = useWaveformPeaks(audioUrl, maxTime);
  const xOf = (time: number) => padL + (time / maxTime) * plotW;
  const yOf = (score: number) => padT + (1 - clamp01(score)) * plotH;

  const markerKeyDown = (event: KeyboardEvent<SVGRectElement>, item: TimelineWindowItem) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(item);
    }
  };

  return (
    <section className="cb-card p-4 sm:p-5" data-testid="window-timeline-card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-base font-medium text-cb-text">Window timeline</h2>
          <p className="mt-1 text-sm text-cb-muted">
            CloakBreak score colors each 2.5 s speech window. Silent windows are hollow.
          </p>
        </div>
        {items.length > 0 ? (
          <p className="text-sm text-cb-muted">
            {canPlayAudio ? "Click a scored window to select it and play that audio." : "Click a scored window to select it."}
          </p>
        ) : null}
      </div>
      {metaText ? (
        <p className="mt-3 rounded-lg border border-cb-border bg-cb-bg px-3 py-2 text-sm text-cb-secondary" data-testid="timeline-meta">
          {metaText}
        </p>
      ) : null}

      <div ref={containerRef} className="mt-4 w-full overflow-x-auto overflow-y-hidden" data-testid="window-timeline">
        {items.length === 0 ? (
          <div className="flex min-h-28 items-center justify-center rounded-lg border border-dashed border-cb-border bg-cb-bg px-4 text-center text-sm text-cb-muted">
            {emptyText ?? "Nothing analyzed yet. Scored windows will appear here."}
          </div>
        ) : (
          <svg width={contentWidth} height={height} viewBox={`0 0 ${contentWidth} ${height}`} role="group" aria-label="Per-window CloakBreak scores">
            {(waveformPeaks ? [] : [0, 0.5, 1]).map((grid) => {
              const y = yOf(grid);
              return (
                <g key={grid}>
                  <line x1={padL} x2={padL + plotW} y1={y} y2={y} stroke="#7d8994" strokeOpacity="0.28" />
                  <text x={6} y={y + 4} fontSize="11" fill="#7d8994" fontFamily="var(--font-mono)">
                    {grid.toFixed(1)}
                  </text>
                </g>
              );
            })}
            <line
              x1={padL}
              x2={padL + plotW}
              y1={yOf(threshold)}
              y2={yOf(threshold)}
              stroke="#b6c0ca"
              strokeOpacity="0.9"
              strokeDasharray="5 5"
            />
            <text
              x={padL + plotW}
              y={Math.max(12, yOf(threshold) - 10)}
              textAnchor="end"
              fontSize="11"
              fill="#7d8994"
              fontFamily="var(--font-mono)"
            >
              threshold {formatScore(threshold, 4)}
            </text>
            {[0, maxTime / 2, maxTime].map((tick, index) => {
              const x = xOf(tick);
              return (
                <text
                  key={`${tick}-${index}`}
                  x={x}
                  y={height - 8}
                  textAnchor={index === 0 ? "start" : index === 2 ? "end" : "middle"}
                  fontSize="11"
                  fill="#7d8994"
                  fontFamily="var(--font-mono)"
                >
                  {timeLabel(tick)}
                </text>
              );
            })}
            {items.map((item) => {
              const x1 = xOf(item.start);
              const x2 = Math.max(x1 + 2, xOf(item.end));
              const segmentW = Math.max(2, x2 - x1 - 2);
              const selected = item.key === selectedKey;
              const scored = item.isSpeech && item.spoofProb != null;
              const score = scored ? item.spoofProb : null;
              const color = scoreColor(score, threshold);
              const barTop = score == null ? baseY - 8 : yOf(score);
              const barHeight = score == null ? 8 : Math.max(2, baseY - yOf(score));
              return (
                <g key={item.key}>
                  <rect
                    x={x1 + 1}
                    y={waveformPeaks ? padT : barTop}
                    width={segmentW}
                    height={waveformPeaks ? plotH : barHeight}
                    rx={2}
                    fill={scored ? color : "#0a0c0f"}
                    stroke={selected ? "#f3f6f8" : scored ? color : "#7d8994"}
                    strokeDasharray={scored ? undefined : "3 3"}
                    strokeWidth={selected ? 2 : 1}
                    opacity={waveformPeaks ? (scored ? 0.16 : 0.35) : scored ? 0.95 : 0.8}
                  />
                  <rect
                    x={x1}
                    y={padT}
                    width={Math.max(minWindowPx / 2, x2 - x1)}
                    height={plotH}
                    fill="transparent"
                    role="button"
                    tabIndex={0}
                    aria-label={`Select window starting at ${timeLabel(item.start)}${score != null ? `, score ${formatScore(score)}` : ", silent"}`}
                    data-testid={`window-marker-${item.rowIndex}`}
                    className="cursor-pointer"
                    onClick={() => onSelect(item)}
                    onKeyDown={(event) => markerKeyDown(event, item)}
                  >
                    <title>{`Window ${timeLabel(item.start)}${score != null ? `: CloakBreak ${formatScore(score)}` : ": silent"}`}</title>
                  </rect>
                </g>
              );
            })}
            {waveformPeaks ? (
              <g data-testid="waveform-strip" pointerEvents="none">
                <line x1={padL} x2={padL + plotW} y1={padT + plotH / 2} y2={padT + plotH / 2} stroke="#7d8994" strokeOpacity="0.3" />
                {waveformPeaks.map((peak, index) => {
                  const centerTime = ((index + 0.5) / waveformPeaks.length) * maxTime;
                  const owner = items.find((item) => centerTime >= item.start && centerTime < item.end) ?? items[items.length - 1];
                  const scored = owner?.isSpeech && owner.spoofProb != null;
                  const color = scoreColor(scored ? owner.spoofProb : null, threshold);
                  const amplitude = clamp01(peak);
                  const barHeight = Math.max(1.5, amplitude * (plotH - 12));
                  const barWidth = Math.max(1, (plotW / waveformPeaks.length) * 0.68);
                  return (
                    <rect
                      key={index}
                      x={xOf(centerTime) - barWidth / 2}
                      y={padT + (plotH - barHeight) / 2}
                      width={barWidth}
                      height={barHeight}
                      rx={0.8}
                      fill={color}
                      opacity={scored ? 0.92 : 0.5}
                    />
                  );
                })}
              </g>
            ) : null}
          </svg>
        )}
      </div>

      {items.length > 0 ? (
        <details className="mt-4 rounded-lg border border-cb-border bg-cb-bg" data-testid="window-details">
          <summary className="flex min-h-11 cursor-pointer list-none items-center px-4 py-2 text-sm font-medium text-cb-text [&::-webkit-details-marker]:hidden">
            Window details
            <span className="ml-auto text-cb-muted" aria-hidden="true">›</span>
          </summary>
          <div className="mx-4 mb-4 overflow-hidden rounded-lg border border-cb-border" data-testid="window-rows">
            <div className="grid grid-cols-[minmax(4.5rem,1fr)_4.5rem_4.5rem_minmax(3.8rem,0.8fr)] gap-2 bg-cb-surface px-3 py-2 text-xs font-medium text-cb-muted sm:grid-cols-[minmax(6rem,1.2fr)_6rem_6rem_minmax(5rem,0.8fr)]">
              <span>Window</span>
              <span className="text-right">CloakBreak</span>
              <span className="text-right">Legacy</span>
              <span className="text-right">State</span>
            </div>
            <div className="max-h-72 overflow-y-auto bg-cb-bg">
              {items.map((item) => {
                const spoofProb = item.spoofProb;
                const scored = item.isSpeech && spoofProb != null;
                const selected = item.key === selectedKey;
                const state = scored && spoofProb != null ? (spoofProb >= threshold ? "Over" : "Under") : "Silent";
                const body = (
                  <>
                    <span className="font-mono tabular-nums text-cb-secondary">
                      {item.start.toFixed(1)}–{item.end.toFixed(1)} s
                    </span>
                    <span className={`text-right font-mono font-medium tabular-nums ${scoreColorClass(item.spoofProb, threshold)}`} data-testid={`window-cb-${item.rowIndex}`}>
                      {formatScore(item.spoofProb)}
                    </span>
                    <span className={`text-right font-mono font-medium tabular-nums ${scoreColorClass(item.baselineProb, 0.5)}`} data-testid={`window-legacy-${item.rowIndex}`}>
                      {formatScore(item.baselineProb)}
                    </span>
                    <span className="text-right text-xs font-medium text-cb-muted">{state}</span>
                  </>
                );
                const gridClass = "grid w-full min-w-0 grid-cols-[minmax(4.5rem,1fr)_4.5rem_4.5rem_minmax(3.8rem,0.8fr)] items-center gap-2 px-3 py-2 text-left text-xs sm:grid-cols-[minmax(6rem,1.2fr)_6rem_6rem_minmax(5rem,0.8fr)] sm:text-sm";
                if (!scored) {
                  return (
                    <div key={item.key} className={`${gridClass} min-h-11 border-t border-cb-border text-cb-muted`} data-testid={`window-row-${item.rowIndex}`}>
                      {body}
                    </div>
                  );
                }
                return (
                  <button
                    key={item.key}
                    type="button"
                    className={`${gridClass} min-h-11 border-t border-cb-border transition-colors duration-150 hover:bg-cb-surface ${selected ? "bg-cb-surface" : "bg-cb-bg"}`}
                    data-testid={`window-row-${item.rowIndex}`}
                    aria-pressed={selected}
                    onClick={() => onSelect(item)}
                  >
                    {body}
                  </button>
                );
              })}
            </div>
          </div>
        </details>
      ) : null}
    </section>
  );
}
