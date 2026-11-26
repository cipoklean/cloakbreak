import { useRef, useState, type ChangeEvent } from "react";
import type { JobProgress } from "../types";
import { Button } from "./Button";
import { JobStatusNote } from "./JobStatus";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const UPLOAD_EXTENSIONS = [".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"];

interface UploadLinkPanelProps {
  inFlight: boolean;
  jobProgress: JobProgress | null;
  analysisError: string | null;
  onAnalyzeUpload: (file: File, objectUrl: string) => void;
  onAnalyzeLink: (url: string) => void;
}

export function UploadLinkPanel({ inFlight, jobProgress, analysisError, onAnalyzeUpload, onAnalyzeLink }: UploadLinkPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [linkUrl, setLinkUrl] = useState("");
  const [clientError, setClientError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const disabledTitle = inFlight ? "Wait for the current analysis to finish" : undefined;

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setClientError(null);
    setSelectedFile(file);
    setObjectUrl(file ? window.URL.createObjectURL(file) : null);
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!UPLOAD_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
      setClientError("Unsupported file type. Use one of: .wav, .mp3, .m4a, .mp4, .webm, .ogg");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setClientError("File too large (max 50 MB)");
    }
  };

  const submitUpload = () => {
    setClientError(null);
    if (!selectedFile) {
      setClientError("Choose an audio or video file first.");
      return;
    }
    const lower = selectedFile.name.toLowerCase();
    if (!UPLOAD_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
      setClientError("Unsupported file type. Use one of: .wav, .mp3, .m4a, .mp4, .webm, .ogg");
      return;
    }
    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setClientError("File too large (max 50 MB)");
      return;
    }
    const url = objectUrl ?? window.URL.createObjectURL(selectedFile);
    setObjectUrl(url);
    onAnalyzeUpload(selectedFile, url);
  };

  const submitLink = () => {
    setClientError(null);
    const trimmed = linkUrl.trim();
    if (!trimmed) {
      setClientError("Paste a link first.");
      return;
    }
    onAnalyzeLink(trimmed);
  };

  return (
    <div className="space-y-5" data-testid="upload-panel">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-base font-semibold text-slate-950">Upload or paste a link</h2>
          <span className="rounded-full border border-amber-300 bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-800">Experimental</span>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Experimental. Trained on speech from a few voice cloners. Noise lowers measured accuracy; music and unseen cloners beyond Kokoro are untested.
        </p>
        <p className="mt-1 text-sm text-slate-600">Analysis covers at most the first 60 seconds.</p>
      </div>

      <JobStatusNote progress={jobProgress && (jobProgress.kind === "upload" || jobProgress.kind === "url") ? jobProgress : null} />
      {(clientError || analysisError) && jobProgress == null ? (
          <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-800" role="alert" data-testid="analysis-error">
            {clientError ?? analysisError}
          </p>
        ) : null}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <section className="rounded-xl border border-[#d5e2f0] bg-slate-50 p-4">
          <label htmlFor="upload-file-input" className="block text-sm font-semibold text-slate-950">
            Choose an audio or video file
          </label>
          <input
            ref={fileInputRef}
            id="upload-file-input"
            type="file"
            accept=".wav,.mp3,.m4a,.mp4,.webm,.ogg"
            data-testid="upload-file-input"
            className="mt-2 block min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 file:mr-3 file:min-h-9 file:rounded-lg file:border file:border-slate-300 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-slate-800 hover:file:bg-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0369a1] focus-visible:ring-offset-2"
            onChange={onFileChange}
          />
          <p className="mt-2 text-xs text-slate-600">Accepted: .wav, .mp3, .m4a, .mp4, .webm, .ogg up to 50 MB.</p>
          {selectedFile ? (
            <p className="mt-2 break-words font-mono text-xs text-slate-700" data-testid="selected-file-name">
              {selectedFile.name} · {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          ) : null}
          <Button
            variant="primary"
            className="mt-3"
            data-testid="analyze-upload-button"
            disabled={inFlight}
            title={disabledTitle}
            onClick={submitUpload}
          >
            {inFlight && jobProgress?.kind === "upload" ? "Analyzing…" : "Analyze clip"}
          </Button>
        </section>

        <section className="rounded-xl border border-[#d5e2f0] bg-slate-50 p-4">
          <label htmlFor="link-input" className="block text-sm font-semibold text-slate-950">
            Paste a video or audio link
          </label>
          <input
            id="link-input"
            type="url"
            inputMode="url"
            autoComplete="url"
            placeholder="https://example.com/video"
            value={linkUrl}
            data-testid="link-input"
            className="mt-2 block min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0369a1] focus-visible:ring-offset-2"
            onChange={(event) => setLinkUrl(event.target.value)}
          />
          <p className="mt-2 text-xs text-slate-600">The server downloads the media and analyzes at most the first 60 seconds. Audio playback is not available for link results.</p>
          <Button
            variant="primary"
            className="mt-3"
            data-testid="analyze-link-button"
            disabled={inFlight}
            title={disabledTitle}
            onClick={submitLink}
          >
            {inFlight && jobProgress?.kind === "url" ? "Analyzing…" : "Analyze link"}
          </Button>
        </section>
      </div>
    </div>
  );
}
