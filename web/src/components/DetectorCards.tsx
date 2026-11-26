import { clamp01, formatScore, scoreColor, verdictWord } from "../utils";

interface DetectorCardProps {
  title: string;
  score: number | null;
  threshold: number;
  unavailable?: boolean;
  caption: string;
  testId: string;
}

function DetectorCard({ title, score, threshold, unavailable = false, caption, testId }: DetectorCardProps) {
  const color = unavailable ? "#64748b" : scoreColor(score, threshold);
  const width = unavailable || score == null ? 0 : clamp01(score) * 100;
  const verdict = unavailable ? "Not loaded" : verdictWord(score, threshold);
  return (
    <article className="card rounded-2xl border border-[#d5e2f0] bg-white p-5 shadow-sm" data-testid={testId}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-slate-950">{title}</h2>
        <div className={`text-sm font-semibold ${verdict === "Clone" ? "text-red-700" : verdict === "Real" ? "text-green-700" : "text-slate-500"}`}>
          {verdict}
        </div>
      </div>
      <div className="mt-4">
        <div className="font-mono text-5xl font-semibold leading-none tabular-nums text-slate-950" data-testid={`${testId}-score`}>
          {unavailable ? "N/A" : formatScore(score)}
        </div>
        <div className="mt-1 text-xs text-slate-600">{caption}</div>
      </div>
      <div className="relative mt-5 h-3 overflow-hidden rounded-full border border-[#d5e2f0] bg-slate-100" aria-hidden="true">
        <div
          className="h-full rounded-full transition-[width,background-color] duration-200 ease-out"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
        <div
          className="absolute bottom-0 top-0 border-l-2 border-dashed border-slate-700/70"
          style={{ left: `${clamp01(threshold) * 100}%` }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] tabular-nums text-slate-600">
        <span>0</span>
        <span>threshold {formatScore(threshold, threshold === 0.5 ? 1 : 4)}</span>
        <span>1</span>
      </div>
    </article>
  );
}

interface DetectorCardsProps {
  cloakScore: number | null;
  legacyScore: number | null;
  threshold: number;
  baselineLoaded: boolean;
}

export function DetectorCards({ cloakScore, legacyScore, threshold, baselineLoaded }: DetectorCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2" data-testid="detector-cards">
      <DetectorCard
        title="CloakBreak"
        score={cloakScore}
        threshold={threshold}
        caption="window spoof score"
        testId="detector-card-cloakbreak"
      />
      <DetectorCard
        title="Legacy detector (trained on replay attacks)"
        score={legacyScore}
        threshold={0.5}
        unavailable={!baselineLoaded}
        caption="baseline spoof score"
        testId="detector-card-legacy"
      />
    </div>
  );
}
