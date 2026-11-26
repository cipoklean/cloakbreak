import { clamp01, formatScore, scoreColor } from "../utils";

interface AttentionHeatmapProps {
  focus: number[] | null;
  score: number | null;
  threshold: number;
  windowLabel?: string | null;
}

export function AttentionHeatmap({ focus, score, threshold, windowLabel }: AttentionHeatmapProps) {
  const bins = focus && focus.length ? focus.slice(0, 32) : null;
  const color = scoreColor(score, threshold);
  return (
    <section className="card rounded-2xl border border-[#d5e2f0] bg-white p-5 shadow-sm" data-testid="attention-card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-slate-950">Model attention focus</h2>
        {windowLabel ? <p className="font-mono text-xs tabular-nums text-slate-600">{windowLabel}</p> : null}
      </div>
      <div className="mt-4 flex h-40 items-end gap-[3px] rounded-xl border border-[#d5e2f0] bg-slate-100 p-3" data-testid="attention-heatmap" aria-label="Attention values for the selected window">
        {bins ? (
          bins.map((value, index) => {
            const clipped = clamp01(Number.isFinite(value) ? value : 0);
            return (
              <div
                key={index}
                className="min-w-1 flex-1 rounded-sm transition-[height,background-color] duration-150"
                style={{
                  height: `${Math.max(3, clipped * 100)}%`,
                  backgroundColor: score == null ? "#64748b" : color,
                  opacity: 0.35 + 0.65 * clipped,
                }}
                title={`bin ${index + 1}: ${clipped.toFixed(2)}`}
              />
            );
          })
        ) : (
          <div className="flex h-full w-full items-center justify-center text-center text-sm text-slate-600">
            Select a scored window to see its attention values.
          </div>
        )}
      </div>
      <p className="mt-3 text-sm text-slate-600">
        Where the model attended within the 2.5 s window. It shows model focus only, not which part is fake.
        {score != null ? ` Selected window score ${formatScore(score)}.` : ""}
      </p>
    </section>
  );
}
