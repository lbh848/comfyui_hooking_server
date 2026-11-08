import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/setting/",
  server: { host: "127.0.0.1", port: 8201, strictPort: true },
  preview: { host: "127.0.0.1", port: 8201, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true },
});
