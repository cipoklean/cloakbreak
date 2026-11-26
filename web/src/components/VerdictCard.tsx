import { formatScore } from "../utils";

interface VerdictCardProps {
  state: "real" | "clone" | "listening" | "analyzing";
  score: number | null;
  reason: string;
  freshness?: string | null;
  liveWarning?: string | null;
}

const stateStyles = {
  real: {
    card: "border-green-300 bg-green-100 text-green-950",
    label: "REAL VOICE",
  },
  clone: {
    card: "border-red-300 bg-red-100 text-red-950",
    label: "LIKELY CLONE",
  },
  listening: {
    card: "border-slate-300 bg-slate-200 text-slate-800",
    label: "LISTENING",
  },
  analyzing: {
    card: "border-slate-300 bg-slate-200 text-slate-800",
    label: "ANALYZING",
  },
} as const;

export function VerdictCard({ state, score, reason, freshness, liveWarning }: VerdictCardProps) {
  const styles = stateStyles[state];
  return (
    <section
      className={`rounded-2xl border p-5 shadow-sm transition-colors duration-150 sm:p-6 ${styles.card}`}
      aria-live="polite"
      aria-busy={state === "analyzing"}
      data-testid="verdict-card"
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-[clamp(2.5rem,9vw,4.5rem)] font-semibold leading-none tracking-tight" data-testid="verdict-title">
            {styles.label}
          </div>
          <p className="mt-3 max-w-3xl text-sm font-medium sm:text-base" data-testid="verdict-reason">
            {reason}
          </p>
          {freshness ? (
            <p className="mt-2 font-mono text-sm tabular-nums" data-testid="verdict-freshness">
              {freshness}
            </p>
          ) : null}
        </div>
        <div className="shrink-0 rounded-xl border border-current/20 bg-white/50 px-4 py-3 text-left sm:text-right">
          <div className="text-xs font-semibold uppercase tracking-wide opacity-75">Key score</div>
          <div className="font-mono text-4xl font-semibold tabular-nums" data-testid="verdict-score">
            {formatScore(score)}
          </div>
        </div>
      </div>
      {liveWarning ? (
        <p className="mt-4 border-t border-slate-950/15 pt-3 text-sm leading-relaxed" data-testid="live-warning">
          {liveWarning}
        </p>
      ) : null}
    </section>
  );
}
