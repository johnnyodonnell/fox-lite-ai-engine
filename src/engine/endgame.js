// Endgame database — runtime API.
//
// Phase A: an in-memory Map populated lazily by alpha-beta inside
// doubleDummy.js. Cold per session; persists only within a single page-load.
// Phase B will replace the backing store with a shipped binary blob loaded
// at startup, without changing this surface.
//
// Values stored here are the exact double-dummy signed margin (-6..+6) in
// the **to-move player's frame** at the cached state. Callers convert to
// whatever frame they want (e.g., doubleDummy.solve negates when the cached
// state's to-move player differs from the root mover).

import { canonKey, tricksRemaining } from './endgame-canon.js'

// Phase A coverage bound: any state with more than this many tricks
// remaining is outside the DB. lookup() returns null for those; put() is
// a no-op. Phase B will pin the final bound after a sizing pass.
const N_MAX = 10

const table = new Map()
let hits = 0
let misses = 0

export function covers(world) {
  return tricksRemaining(world) <= N_MAX
}

export function lookup(world) {
  if (!covers(world)) {
    misses++
    return null
  }
  const key = canonKey(world)
  if (table.has(key)) {
    hits++
    return table.get(key)
  }
  misses++
  return null
}

export function put(world, value) {
  if (!covers(world)) return
  const key = canonKey(world)
  table.set(key, value)
}

export function stats() {
  return { hits, misses, size: table.size }
}

export function resetStats() {
  hits = 0
  misses = 0
}
