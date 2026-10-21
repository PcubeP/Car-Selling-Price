import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // onnxruntime-web ships wasm binaries that Vite shouldn't try to pre-bundle.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
