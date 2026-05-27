// Network parity check (JS side): run the JS forward pass on the inputs
// dumped by network_parity_dump.py and assert agreement with the Python
// reference outputs. Both sides do float64 arithmetic, so any difference
// above ~1e-12 indicates a real implementation bug rather than precision
// drift.
//
// Run (after network_parity_dump.py):
//   node training/scripts/network_parity_check.mjs

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { forward } from '../../src/engine/nn.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, '..', '..')

const FORWARD_TOL = 1e-9 // generous; observed parity in tic-tac-toe is ~1e-15

const weightsPath = path.join(repoRoot, 'src/engine/weights.json')
const dumpPath = path.join(here, '..', 'network_parity_expected.json')

if (!fs.existsSync(weightsPath)) {
  console.error(`missing ${weightsPath} — run export_weights.py first`)
  process.exit(2)
}
if (!fs.existsSync(dumpPath)) {
  console.error(`missing ${dumpPath} — run network_parity_dump.py first`)
  process.exit(2)
}

const weights = JSON.parse(fs.readFileSync(weightsPath, 'utf8'))
const dump = JSON.parse(fs.readFileSync(dumpPath, 'utf8'))

function maxAbsDiff(a, b) {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

let failures = 0
let worstLogit = 0
let worstValue = 0

for (let i = 0; i < dump.cases.length; i++) {
  const c = dump.cases[i]
  const { policyLogits, value } = forward(weights, c.input)
  const dl = maxAbsDiff(policyLogits, c.policyLogits)
  const dv = Math.abs(value - c.value)
  worstLogit = Math.max(worstLogit, dl)
  worstValue = Math.max(worstValue, dv)
  if (dl > FORWARD_TOL || dv > FORWARD_TOL) {
    console.log(
      `FAIL case[${i}]  Δlogit=${dl.toExponential(2)}  Δvalue=${dv.toExponential(2)}`
    )
    failures++
  }
}

console.log(`\nworst Δlogit=${worstLogit.toExponential(2)}  worst Δvalue=${worstValue.toExponential(2)}`)
if (failures === 0) {
  console.log(`PARITY OK — JS forward pass matches Python (${dump.cases.length} cases).`)
  process.exit(0)
} else {
  console.log(`PARITY FAILED — ${failures}/${dump.cases.length} cases mismatched.`)
  process.exit(1)
}
