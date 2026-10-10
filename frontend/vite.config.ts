import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The SPA is served same-origin behind nginx in production (static files at /,
// API proxied at /api). In dev, Vite proxies /api, /health and /openapi.json to
// a locally running FastAPI so the frontend talks to the real backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/openapi.json": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    // plotly.js is large; raise the warning ceiling rather than code-split a
    // dashboard whose first paint needs the chart library anyway.
    chunkSizeWarningLimit: 4000,
  },
});
