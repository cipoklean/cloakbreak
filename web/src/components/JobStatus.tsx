import type { JobProgress } from "../types";
import { jobProgressText } from "../utils";

export function JobStatusNote({ progress }: { progress: JobProgress | null }) {
  const text = jobProgressText(progress);
  if (!text) return null;
  return (
    <p className="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-slate-800" role="status" data-testid="job-status">
      {text}
    </p>
  );
}
