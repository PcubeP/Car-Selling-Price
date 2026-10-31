import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // onnxruntime-web exposes a "use-extern-wasm" export condition whose entry
    // loads the wasm from env.wasm.wasmPaths (our CDN) instead of emitting a
    // ~28 MB binary into dist via `new URL(..., import.meta.url)`. We set
    // wasmPaths to the CDN, so the bundled copy is dead weight. This condition is
    // namespaced to onnxruntime-web, so it doesn't affect other packages.
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // onnxruntime-web ships wasm binaries that Vite shouldn't try to pre-bundle.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
