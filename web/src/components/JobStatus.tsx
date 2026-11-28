import type { JobProgress } from "../types";
import { jobProgressText } from "../utils";

export function JobStatusNote({ progress }: { progress: JobProgress | null }) {
  const text = jobProgressText(progress);
  if (!text) return null;
  return (
    <p className="rounded-lg border border-cb-border bg-cb-bg px-4 py-3 text-sm font-medium text-cb-secondary" role="status" data-testid="job-status">
      {text}
    </p>
  );
}
