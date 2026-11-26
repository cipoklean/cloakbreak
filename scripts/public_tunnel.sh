#!/usr/bin/env bash
# Expose the local CloakBreak demo (port 8765) through a Cloudflare quick tunnel.
#
# A quick tunnel gives a temporary public https://*.trycloudflare.com URL with
# no account or signup. Anyone with the URL can use the demo, so:
#   - the clone endpoint MUST stay off: this script refuses to run if
#     ENABLE_CLONE_ENDPOINT=true is set in this shell, and the server itself
#     keeps it off by default;
#   - live-mic audio of whoever uses it is sent to this machine — say so to
#     anyone you share the URL with;
#   - stop the tunnel (Ctrl+C) when the demo is over.
set -euo pipefail

PORT="${CLOAKBREAK_PORT:-8765}"

if [[ "${ENABLE_CLONE_ENDPOINT:-false}" == "true" ]]; then
  echo "[tunnel] REFUSING to start: ENABLE_CLONE_ENDPOINT=true." >&2
  echo "[tunnel] The private clone endpoint must never be exposed publicly." >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[tunnel] cloudflared is not installed. Install it once with:
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
  chmod +x /tmp/cloudflared && sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
  (ARM machines: use cloudflared-linux-arm64 instead)
MSG
  exit 1
fi

if ! curl -sS -o /dev/null --max-time 5 "http://localhost:${PORT}/api/health"; then
  echo "[tunnel] nothing answers at http://localhost:${PORT}/api/health" >&2
  echo "[tunnel] start the app first:" >&2
  echo "  CLOAKBREAK_BACKEND=onnx CLOAKBREAK_MODEL_PATH=artifacts/session2/onnx/cloakbreak_fp32.onnx \\" >&2
  echo "  .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port ${PORT}" >&2
  exit 1
fi

echo "[tunnel] sharing http://localhost:${PORT} — the public URL appears below ('trycloudflare.com')."
echo "[tunnel] clone endpoint: off. Stop with Ctrl+C when the demo is over."
exec cloudflared tunnel --url "http://localhost:${PORT}" --no-autoupdate
