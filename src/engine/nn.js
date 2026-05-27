// Hand-written forward pass for the policy/value network. Mirrors
// training/alphazero/network.py exactly:
//   - each trunk layer is a Linear followed by ReLU
//   - the policy head is Linear (raw logits)
//   - the value head is Linear then tanh
//
// `weights` is the parsed weights.json. The function is pure so the same
// module loads in the browser app, in the Node parity check, and in any
// future server-side use.

// Linear layer: `w` is row-major [out][in], `b` is [out]. Returns a length-out
// array. Loop order matches export_weights.py's serialization.
export function linear(input, w, b) {
  const out = new Array(w.length)
  for (let o = 0; o < w.length; o++) {
    const row = w[o]
    let sum = b[o]
    for (let i = 0; i < row.length; i++) {
      sum += row[i] * input[i]
    }
    out[o] = sum
  }
  return out
}

// Forward pass. Returns { policyLogits: number[], value: number }.
export function forward(weights, input) {
  let x = input
  for (const layer of weights.trunk) {
    x = linear(x, layer.w, layer.b).map((v) => (v > 0 ? v : 0)) // ReLU
  }
  const policyLogits = linear(x, weights.policyHead.w, weights.policyHead.b)
  const valueRaw = linear(x, weights.valueHead.w, weights.valueHead.b)[0]
  return { policyLogits, value: Math.tanh(valueRaw) }
}
