import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/small_vlm_sop_check/apps/frontend_dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        assetFileNames: "assets/app.[ext]",
      },
    },
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8501" },
  },
});
