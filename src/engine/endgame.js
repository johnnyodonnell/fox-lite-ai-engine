// Endgame database — runtime API.
//
// Two-tier backing store:
//
//   1. Blob layer (read-only).  endgame-data.bin is shipped with the app and
//      loaded asynchronously at module init. Holds exact double-dummy values
//      for every canonical state with tricks_remaining ≤ N_MAX_BLOB (= 6).
//      Lookup is a fixed-stride binary search on packed bytes.
//
//   2. Map layer (read/write).  In-memory Map populated lazily by alpha-beta
//      via put() for states with tricks_remaining ≤ N_MAX_MAP. Acts as a
//      session-level memoization cache for states the blob doesn't cover
//      (and as the sole backing store before the blob finishes loading, or
//      in Node where the blob isn't loaded at all).
//
// Values are int8 signed margins (-6..+6) in the **to-move player's frame**
// at the cached state. Callers convert to whatever frame they want
// (doubleDummy.solve negates when the cached state's to-move player differs
// from the root mover).

import { canonKey, tricksRemaining } from './endgame-canon.js'
import {
  FILE_HEADER_BYTES,
  BLOB_MAGIC,
  BLOB_VERSION,
  keyByteWidth,
  entryByteWidth,
  locationBytesForNMax,
  packCanonKey,
  lookupPacked,
} from './endgame-blob.js'

const N_MAX_MAP = 10

const table = new Map()
let hits = 0
let misses = 0

// Blob state. Populated asynchronously in browser environments; stays at
// defaults (and lookups fall through to the Map) in Node and pre-load.
let blobBytes = null // Uint8Array view of the entry table (no file header)
let blobNMax = 0
let blobKeyWidth = 0
let blobEntryWidth = 0
let blobLocBytes = 0
let blobEntryCount = 0
let keyScratch = null // reused Uint8Array for packing query keys

function installBlob(buf) {
  const dv = new DataView(buf)
  const magic = dv.getUint32(0, true)
  if (magic !== BLOB_MAGIC) throw new Error('endgame blob magic mismatch')
  const version = dv.getUint16(4, true)
  if (version !== BLOB_VERSION) throw new Error(`endgame blob version ${version} not supported`)
  blobNMax = dv.getUint32(8, true)
  blobEntryCount = dv.getUint32(12, true)
  blobKeyWidth = keyByteWidth(blobNMax)
  blobEntryWidth = entryByteWidth(blobNMax)
  blobLocBytes = locationBytesForNMax(blobNMax)
  keyScratch = new Uint8Array(blobKeyWidth)
  blobBytes = new Uint8Array(buf, FILE_HEADER_BYTES)
}

// Browser-only auto-load. The build script (Node) must NOT auto-load — if
// it did, solve() would short-circuit via blob hits and skip repopulating
// the Map. Node scripts that want to exercise the blob layer call
// _installBlobBuffer(buffer) explicitly.
if (typeof window !== 'undefined') {
  const url = new URL('./endgame-data.bin', import.meta.url).href
  fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${url} → ${r.status}`)
      return r.arrayBuffer()
    })
    .then((buf) => {
      installBlob(buf)
      console.log(
        `[endgame] blob loaded: N_MAX=${blobNMax}, ` +
          `${blobEntryCount.toLocaleString()} entries, ` +
          `${(buf.byteLength / 1024 / 1024).toFixed(2)} MB`
      )
    })
    .catch((err) => {
      console.warn('[endgame] blob load failed:', err)
    })
}

// Build-time / test-time hook: install a blob buffer manually (Node).
export function _installBlobBuffer(buf) {
  installBlob(buf)
}

// Test-time hook: enable/disable the blob layer at runtime. When disabled,
// lookup() ignores the blob and only consults the Map. Used by strength
// A/B tests to compare with-blob vs without-blob behavior in one process.
let blobEnabled = true
export function _setBlobEnabled(enabled) {
  blobEnabled = enabled
}

// Test-time hook: clear the Map layer so a verification step can isolate
// blob behavior from Map fallback.
export function _clearMap() {
  table.clear()
}

// Test-time hook: query the blob directly with no Map fallback. Returns
// the int8 value or null if the blob doesn't cover this state.
export function _lookupBlobOnly(world) {
  if (blobBytes === null) return null
  if (tricksRemaining(world) > blobNMax) return null
  const key = canonKey(world)
  keyScratch.fill(0)
  packCanonKey(key, keyScratch, 0, blobLocBytes)
  return lookupPacked(blobBytes, keyScratch, blobKeyWidth, blobEntryWidth, blobEntryCount)
}

export function covers(world) {
  return tricksRemaining(world) <= N_MAX_MAP
}

export function lookup(world) {
  const tr = tricksRemaining(world)
  const inBlobBand = blobEnabled && blobBytes !== null && tr <= blobNMax
  const inMapBand = tr <= N_MAX_MAP

  if (!inBlobBand && !inMapBand) {
    misses++
    return null
  }

  const key = canonKey(world)

  if (inBlobBand) {
    keyScratch.fill(0)
    packCanonKey(key, keyScratch, 0, blobLocBytes)
    const v = lookupPacked(blobBytes, keyScratch, blobKeyWidth, blobEntryWidth, blobEntryCount)
    if (v !== null) {
      hits++
      return v
    }
  }

  if (inMapBand && table.has(key)) {
    hits++
    return table.get(key)
  }

  misses++
  return null
}

export function put(world, value) {
  if (tricksRemaining(world) > N_MAX_MAP) return
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

// Build-time only: walk the in-memory table for serialization. Not used by
// any runtime engine code.
export function entries() {
  return table.entries()
}
