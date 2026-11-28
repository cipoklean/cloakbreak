import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import type { JobProgress } from "../types";
import { Button } from "./Button";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const UPLOAD_EXTENSIONS = [".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"];

interface UploadLinkPanelProps {
  inFlight: boolean;
  jobProgress: JobProgress | null;
  uploadError: string | null;
  linkError: string | null;
  onAnalyzeUpload: (file: File, objectUrl: string) => void;
  onAnalyzeLink: (url: string) => void;
}

function validateUpload(file: File): string | null {
  const lower = file.name.toLowerCase();
  if (!UPLOAD_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return "Unsupported file type. Use one of: .wav, .mp3, .m4a, .mp4, .webm, .ogg";
  }
  if (file.size > MAX_UPLOAD_BYTES) return "File too large (max 50 MB)";
  return null;
}

export function UploadLinkPanel({ inFlight, jobProgress, uploadError, linkError, onAnalyzeUpload, onAnalyzeLink }: UploadLinkPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [linkUrl, setLinkUrl] = useState("");
  const [fileError, setFileError] = useState<string | null>(null);
  const [linkClientError, setLinkClientError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const disabledTitle = inFlight ? "Wait for the current analysis to finish" : undefined;

  const acceptFile = (file: File | null) => {
    setFileError(null);
    setSelectedFile(file);
    setObjectUrl(file ? window.URL.createObjectURL(file) : null);
    if (!file) return;
    setFileError(validateUpload(file));
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    acceptFile(event.target.files?.[0] ?? null);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0] ?? null);
  };

  const submitUpload = () => {
    setFileError(null);
    if (!selectedFile) {
      setFileError("Choose an audio or video file first.");
      return;
    }
    const invalid = validateUpload(selectedFile);
    if (invalid) {
      setFileError(invalid);
      return;
    }
    const url = objectUrl ?? window.URL.createObjectURL(selectedFile);
    setObjectUrl(url);
    onAnalyzeUpload(selectedFile, url);
  };

  const submitLink = () => {
    setLinkClientError(null);
    const trimmed = linkUrl.trim();
    if (!trimmed) {
      setLinkClientError("Paste a link first.");
      return;
    }
    onAnalyzeLink(trimmed);
  };

  const shownFileError = fileError ?? (jobProgress == null ? uploadError : null);
  const shownLinkError = linkClientError ?? (jobProgress == null ? linkError : null);

  return (
    <div data-testid="upload-panel">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-base font-medium text-cb-text">Upload or paste a link</h2>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-cb-muted">
        Experimental. Trained on speech from a few voice cloners. Noise lowers measured accuracy; music and unseen cloners beyond Kokoro are untested.
      </p>
      <p className="mt-1 text-sm text-cb-muted">Analysis covers at most the first 60 seconds.</p>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="rounded-lg border border-cb-border bg-cb-bg p-4">
          <label
            htmlFor="upload-file-input"
            className={`flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-6 text-center transition-[border-color,background-color] duration-150 ${
              dragging ? "border-cb-secondary bg-cb-surface" : "border-cb-border bg-cb-bg hover:border-cb-muted"
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                fileInputRef.current?.click();
              }
            }}
          >
            <input
              ref={fileInputRef}
              id="upload-file-input"
              type="file"
              accept=".wav,.mp3,.m4a,.mp4,.webm,.ogg"
              data-testid="upload-file-input"
              className="sr-only"
              tabIndex={-1}
              onChange={onFileChange}
            />
            <span className="text-base font-medium text-cb-text">Drop audio or video here</span>
            <span className="mt-2 text-sm text-cb-muted">or choose a file</span>
            <span className="mt-3 text-xs text-cb-muted">Accepted: .wav, .mp3, .m4a, .mp4, .webm, .ogg up to 50 MB.</span>
            {selectedFile ? (
              <span className="mt-3 max-w-full break-words font-mono text-xs tabular-nums text-cb-secondary" data-testid="selected-file-name">
                {selectedFile.name} · {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </span>
            ) : null}
          </label>
          {shownFileError ? (
            <p className="mt-3 rounded-lg border border-cb-clone px-3 py-2 text-sm font-medium text-cb-clone" role="alert" data-testid="analysis-error">
              {shownFileError}
            </p>
          ) : null}
          <Button
            variant="primary"
            className="mt-4"
            data-testid="analyze-upload-button"
            disabled={inFlight}
            title={disabledTitle}
            onClick={submitUpload}
          >
            {inFlight && jobProgress?.kind === "upload" ? "Analyzing…" : "Analyze clip"}
          </Button>
        </section>

        <section className="rounded-lg border border-cb-border bg-cb-bg p-4">
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="link-input" className="block text-sm font-medium text-cb-text">
              Paste a video or audio link
            </label>
            <span className="rounded-lg border border-cb-border px-2 py-1 text-xs font-medium text-cb-muted">Experimental</span>
          </div>
          <input
            id="link-input"
            type="url"
            inputMode="url"
            autoComplete="url"
            placeholder="https://example.com/video"
            value={linkUrl}
            data-testid="link-input"
            className="cb-input mt-2 block w-full px-3 py-2 text-sm placeholder:text-cb-muted"
            onChange={(event) => setLinkUrl(event.target.value)}
          />
          <p className="mt-2 text-xs leading-relaxed text-cb-muted">The server downloads the media and analyzes at most the first 60 seconds. Audio playback is not available for link results.</p>
          {shownLinkError ? (
            <p className="mt-3 rounded-lg border border-cb-clone px-3 py-2 text-sm font-medium text-cb-clone" role="alert" data-testid="link-analysis-error">
              {shownLinkError}
            </p>
          ) : null}
          <Button
            variant="primary"
            className="mt-4"
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
