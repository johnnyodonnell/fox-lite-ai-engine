// onnxruntime-web session wrapper for the Fox-Lite policy/value net.
// Lazily loads /models/current.onnx on first use, caches the session.
// Tries WebGPU, falls back to WASM.
//
//   await loadModel(url?)         // optional pre-warm
//   await evaluate(inputF32)      // -> { policy: Float32Array(33), value: number }

import * as ort from 'onnxruntime-web'

import { INPUT_SIZE, NUM_CARDS } from './encode.js'

// vite.config.js serves onnxruntime-web's wasm/mjs assets under /onnx-wasm/.
ort.env.wasm.wasmPaths = '/onnx-wasm/'

const DEFAULT_MODEL_URL = '/models/current.onnx'

let sessionPromise = null

export function loadModel(url = DEFAULT_MODEL_URL) {
  if (sessionPromise) return sessionPromise
  sessionPromise = (async () => {
    const providers = []
    if (typeof navigator !== 'undefined' && navigator.gpu) providers.push('webgpu')
    providers.push('wasm')
    try {
      return await ort.InferenceSession.create(url, { executionProviders: providers })
    } catch (err) {
      console.warn('FoxNet: WebGPU failed, falling back to WASM', err)
      return await ort.InferenceSession.create(url, { executionProviders: ['wasm'] })
    }
  })()
  return sessionPromise
}

export async function evaluate(inputFloat32) {
  const session = await loadModel()
  const tensor = new ort.Tensor('float32', inputFloat32, [1, INPUT_SIZE])
  const out = await session.run({ input: tensor })
  const policy = out.policy.data
  if (policy.length !== NUM_CARDS) {
    throw new Error(`FoxNet: expected policy length ${NUM_CARDS}, got ${policy.length}`)
  }
  return { policy, value: out.value.data[0] }
}
