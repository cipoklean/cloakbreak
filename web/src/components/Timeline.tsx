import type { KeyboardEvent } from "react";
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
  emptyText?: string;
  metaText?: string | null;
}

function timeLabel(value: number): string {
  return `${value.toFixed(1)} s`;
}

export function Timeline({ items, threshold, selectedKey, onSelect, canPlayAudio, emptyText, metaText }: TimelineProps) {
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const height = 170;
  const padL = 42;
  const padR = 14;
  const padT = 16;
  const padB = 30;
  const plotW = Math.max(40, width - padL - padR);
  const plotH = height - padT - padB;
  const maxTime = Math.max(2.5, ...items.map((item) => item.end));
  const xOf = (time: number) => padL + (time / maxTime) * plotW;
  const yOf = (score: number) => padT + (1 - clamp01(score)) * plotH;

  const markerKeyDown = (event: KeyboardEvent<SVGCircleElement>, item: TimelineWindowItem) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(item);
    }
  };

  return (
    <section className="card rounded-2xl border border-[#d5e2f0] bg-white p-5 shadow-sm" data-testid="window-timeline-card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-slate-950">Window timeline</h2>
          <p className="mt-1 text-sm text-slate-600">
            CloakBreak score for each 2.5 s speech window. Silent windows are hollow.
          </p>
        </div>
        {items.length > 0 ? (
          <p className="text-sm text-slate-600">
            {canPlayAudio ? "Click a scored window to select it and play that audio." : "Click a scored window to select it."}
          </p>
        ) : null}
      </div>
      {metaText ? (
        <p className="mt-3 rounded-xl border border-[#d5e2f0] bg-slate-50 px-3 py-2 text-sm text-slate-700" data-testid="timeline-meta">
          {metaText}
        </p>
      ) : null}

      <div ref={containerRef} className="mt-4 w-full overflow-hidden" data-testid="window-timeline">
        {items.length === 0 ? (
          <div className="flex min-h-28 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 text-center text-sm text-slate-600">
            {emptyText ?? "Nothing analyzed yet. Scored windows will appear here."}
          </div>
        ) : (
          <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-window CloakBreak scores">
            {[0, 0.5, 1].map((grid) => {
              const y = yOf(grid);
              return (
                <g key={grid}>
                  <line x1={padL} x2={padL + plotW} y1={y} y2={y} stroke="#94a3b8" strokeOpacity="0.35" />
                  <text x={6} y={y + 4} fontSize="11" fill="#475569" fontFamily="ui-monospace, monospace">
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
              stroke="#0f172a"
              strokeOpacity="0.75"
              strokeDasharray="5 5"
            />
            <text
              x={padL + plotW}
              y={Math.max(12, yOf(threshold) - 12)}
              textAnchor="end"
              fontSize="11"
              fill="#475569"
              fontFamily="ui-monospace, monospace"
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
                  fill="#475569"
                  fontFamily="ui-monospace, monospace"
                >
                  {timeLabel(tick)}
                </text>
              );
            })}
            {items.map((item) => {
              const x = xOf(item.start);
              if (!item.isSpeech || item.spoofProb == null) {
                return (
                  <circle key={item.key} cx={x} cy={yOf(0)} r={5.5} fill="#ffffff" stroke="#64748b" strokeWidth={1.8}>
                    <title>{`Silent window starting at ${timeLabel(item.start)}`}</title>
                  </circle>
                );
              }
              const selected = item.key === selectedKey;
              const color = scoreColor(item.spoofProb, threshold);
              return (
                <g key={item.key}>
                  <circle cx={x} cy={yOf(item.spoofProb)} r={selected ? 8 : 6.5} fill={color} stroke={selected ? "#0f172a" : "#ffffff"} strokeWidth={selected ? 2.5 : 1.5} />
                  <circle
                    cx={x}
                    cy={yOf(item.spoofProb)}
                    r={22}
                    fill="transparent"
                    role="button"
                    tabIndex={0}
                    aria-label={`Select window starting at ${timeLabel(item.start)}, score ${formatScore(item.spoofProb)}`}
                    data-testid={`window-marker-${item.rowIndex}`}
                    className="cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0369a1]"
                    onClick={() => onSelect(item)}
                    onKeyDown={(event) => markerKeyDown(event, item)}
                  >
                    <title>{`Window ${timeLabel(item.start)}: CloakBreak ${formatScore(item.spoofProb)}`}</title>
                  </circle>
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {items.length > 0 ? (
        <div className="mt-4 overflow-hidden rounded-xl border border-[#d5e2f0]" data-testid="window-rows">
          <div className="grid grid-cols-[minmax(4.5rem,1fr)_4.5rem_4.5rem_minmax(3.8rem,0.8fr)] gap-2 bg-slate-100 px-3 py-2 text-xs font-semibold text-slate-600 sm:grid-cols-[minmax(6rem,1.2fr)_6rem_6rem_minmax(5rem,0.8fr)]">
            <span>Window</span>
            <span className="text-right">CloakBreak</span>
            <span className="text-right">Legacy</span>
            <span className="text-right">State</span>
          </div>
          <div className="max-h-72 overflow-y-auto bg-white">
            {items.map((item) => {
              const spoofProb = item.spoofProb;
              const scored = item.isSpeech && spoofProb != null;
              const selected = item.key === selectedKey;
              const state = scored && spoofProb != null ? (spoofProb >= threshold ? "Over" : "Under") : "Silent";
              const body = (
                <>
                  <span className="font-mono tabular-nums text-slate-800">
                    {item.start.toFixed(1)}–{item.end.toFixed(1)} s
                  </span>
                  <span className={`text-right font-mono font-semibold tabular-nums ${scoreColorClass(item.spoofProb, threshold)}`} data-testid={`window-cb-${item.rowIndex}`}>
                    {formatScore(item.spoofProb)}
                  </span>
                  <span className={`text-right font-mono font-semibold tabular-nums ${scoreColorClass(item.baselineProb, 0.5)}`} data-testid={`window-legacy-${item.rowIndex}`}>
                    {formatScore(item.baselineProb)}
                  </span>
                  <span className="text-right text-xs font-semibold text-slate-600">{state}</span>
                </>
              );
              const gridClass = "grid w-full min-w-0 grid-cols-[minmax(4.5rem,1fr)_4.5rem_4.5rem_minmax(3.8rem,0.8fr)] items-center gap-2 px-3 py-2 text-left text-xs sm:grid-cols-[minmax(6rem,1.2fr)_6rem_6rem_minmax(5rem,0.8fr)] sm:text-sm";
              if (!scored) {
                return (
                  <div key={item.key} className={`${gridClass} min-h-11 border-t border-slate-200 bg-slate-50 text-slate-500`} data-testid={`window-row-${item.rowIndex}`}>
                    {body}
                  </div>
                );
              }
              return (
                <button
                  key={item.key}
                  type="button"
                  className={`${gridClass} min-h-11 border-t border-slate-200 transition duration-150 hover:bg-sky-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#0369a1] ${selected ? "bg-sky-100" : "bg-white"}`}
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
      ) : null}
    </section>
  );
}
