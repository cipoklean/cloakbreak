import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// @types/node is deliberately not a dependency of this scaffold; the only
// node API the config needs is process.env.
declare const process: { env: Record<string, string | undefined> };

// Dev-server proxy target for the FastAPI backend; override with
// CLOAKBREAK_PROXY_TARGET when the backend is not on the default port.
const proxyTarget = process.env.CLOAKBREAK_PROXY_TARGET ?? "http://localhost:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
  },
  define: {
    // Pass through the build-time API base; the UI falls back to "" (same
    // origin) when it is unset: const API = import.meta.env.VITE_API_BASE ?? ""
    "import.meta.env.VITE_API_BASE": JSON.stringify(process.env.VITE_API_BASE ?? ""),
  },
  server: {
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
      },
      "/ws": {
        target: proxyTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
