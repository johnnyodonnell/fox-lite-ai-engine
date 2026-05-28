// Value-consistency check for src/engine/endgame-data.bin.
//
// For each of N sampled reachable positions with tricks_remaining ≤ N_MAX_BLOB:
//   1. Compute V_truth by running solve() with a fresh empty Map and no blob.
//   2. Install the blob.
//   3. Read V_blob via _lookupBlobOnly (no Map fallback).
//   4. Assert V_truth === V_blob and that V_blob is not null.
//
// Catches any pack/unpack divergence, build/runtime format mismatches, or
// silently-truncated entries. Per the Phase B plan, delete after blob is
// trusted in production.

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  HUMAN,
  BOT,
  createGame,
  legalMoves,
  playCard,
  advanceAfterTrick,
} from '../src/engine/game.js'
import { solve } from '../src/engine/doubleDummy.js'
import { tricksRemaining } from '../src/engine/endgame-canon.js'
import * as endgame from '../src/engine/endgame.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BLOB_PATH = resolve(__dirname, '../src/engine/endgame-data.bin')

const NUM_SAMPLES = 1000
const N_MAX_BLOB_EXPECTED = 6

function step(world, card) {
  let next = playCard(world, card)
  while (next.phase === 'trick-complete') next = advanceAfterTrick(next)
  return next
}

function legalForMover(world) {
  const hand = world.awaiting === HUMAN ? world.humanHand : world.botHand
  return legalMoves(hand, world.ledCard)
}

function collectFromOneGame(out, want) {
  let s = createGame()
  while (s.phase === 'playing' && out.length < want) {
    const tr = tricksRemaining(s)
    if (tr >= 1 && tr <= N_MAX_BLOB_EXPECTED) out.push(s)
    const legal = legalForMover(s)
    s = step(s, legal[Math.floor(Math.random() * legal.length)])
  }
}

function main() {
  console.log('Sampling positions with tricks_remaining ≤ 6...')
  const samples = []
  while (samples.length < NUM_SAMPLES) collectFromOneGame(samples, NUM_SAMPLES)
  samples.length = NUM_SAMPLES
  console.log(`Got ${samples.length} samples.`)

  // Step 1: compute V_truth without any blob installed. Map fills up as
  // solve() recurses, but each top-level call starts from the same state
  // so its V_truth is the true minimax value regardless.
  console.log('Computing V_truth via solve() (blob not installed)...')
  const truths = new Array(samples.length)
  const tStart = Date.now()
  for (let i = 0; i < samples.length; i++) {
    const tt = new Map()
    const { value } = solve(samples[i], -Infinity, Infinity, tt, samples[i].awaiting)
    truths[i] = value
    if ((i + 1) % 200 === 0) {
      console.log(`  ${i + 1}/${samples.length} solved (${Date.now() - tStart} ms)`)
    }
  }
  console.log(`Truths computed in ${Date.now() - tStart} ms.`)

  // Step 2: install blob.
  const buf = readFileSync(BLOB_PATH)
  const arrayBuf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  endgame._installBlobBuffer(arrayBuf)
  endgame._clearMap()
  console.log('Blob installed; Map cleared.')

  // Step 3 + 4: verify blob values.
  let mismatches = 0
  let missingCoverage = 0
  for (let i = 0; i < samples.length; i++) {
    const blobValue = endgame._lookupBlobOnly(samples[i])
    if (blobValue === null) {
      missingCoverage++
      if (missingCoverage <= 3) {
        console.log(`  MISS: sample ${i} (tr=${tricksRemaining(samples[i])}) not in blob`)
      }
      continue
    }
    if (blobValue !== truths[i]) {
      mismatches++
      if (mismatches <= 3) {
        console.log(
          `  MISMATCH: sample ${i} truth=${truths[i]} blob=${blobValue} ` +
            `(tr=${tricksRemaining(samples[i])})`
        )
      }
    }
  }

  console.log('')
  console.log('=== Results ===')
  console.log(`Samples              : ${samples.length}`)
  console.log(`Coverage misses      : ${missingCoverage}`)
  console.log(`Value mismatches     : ${mismatches}`)

  if (mismatches > 0 || missingCoverage > 0) {
    console.log('FAIL: blob is not consistent with solve().')
    process.exit(1)
  }
  console.log('PASS: blob values agree with solve() for every sample.')
}

main()
