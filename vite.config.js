import { createReadStream, readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// onnxruntime-web dynamically imports its wasm/jsep glue (.mjs) at runtime.
// Vite's dev server refuses module imports of files under /public/, so we serve
// them straight from node_modules under a virtual /onnx-wasm/ prefix — for dev
// (middleware) and production (emitted assets).
const ORT_DIR = resolve('node_modules/onnxruntime-web/dist')
const ORT_FILES = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
]

const onnxWasmAssets = {
  name: 'serve-onnx-wasm',
  configureServer(server) {
    server.middlewares.use('/onnx-wasm', (req, res, next) => {
      const filename = req.url.split('?')[0].replace(/^\//, '')
      if (!ORT_FILES.includes(filename)) return next()
      const filepath = resolve(ORT_DIR, filename)
      if (!existsSync(filepath)) return next()
      res.setHeader(
        'Content-Type',
        filename.endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
      )
      createReadStream(filepath).pipe(res)
    })
  },
  generateBundle() {
    for (const file of ORT_FILES) {
      this.emitFile({
        type: 'asset',
        fileName: `onnx-wasm/${file}`,
        source: readFileSync(resolve(ORT_DIR, file)),
      })
    }
  },
}

export default defineConfig({
  plugins: [react(), onnxWasmAssets],
})
