// Phase B retrograde builder for the endgame database.
//
// Subcommands:
//   verify-enum   round-trip check: every enumerated canonical key must
//                 match canonKey on the materialized representative world.
//                 Gate before trusting sizing or build output.
//   size          Pass 1. Count canonical states per depth d in [1, 13].
//                 Prints a table used to auto-pick N_MAX.
//   build         Pass 2. Enumerates depths 1..N_MAX, drives solve() for
//                 each canonical state, serializes the populated DB Map
//                 to src/engine/endgame-data.bin.
//
// Per the Phase B plan, this script is temporary; delete after a stable
// blob ships (or keep if we may rebuild).

import { writeFileSync, existsSync, readFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { HUMAN, BOT } from '../src/engine/game.js'
import { canonKey } from '../src/engine/endgame-canon.js'
import { solve } from '../src/engine/doubleDummy.js'
import * as endgame from '../src/engine/endgame.js'
import {
  FILE_HEADER_BYTES,
  BLOB_MAGIC,
  BLOB_VERSION,
  HEADER_BYTES as KEY_HEADER_BYTES,
  locationBytesForNMax,
  keyByteWidth,
  entryByteWidth,
  packCanonKey,
} from '../src/engine/endgame-blob.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(__dirname, '..')
const BLOB_PATH = resolve(REPO_ROOT, 'src/engine/endgame-data.bin')
const SIZE_CACHE_PATH = resolve(REPO_ROOT, 'scripts/.endgame-size-cache.json')

const TRICKS_PER_ROUND = 13
const RANKS_PER_SUIT = 11
const MAX_TRUMP_IN_PLAY = RANKS_PER_SUIT - 1 // 1 rank is the revealed trump card
const TARGET_BLOB_BYTES = 50 * 1024 * 1024

// ----- Multinomial string generator -----
//
// Yields all distinct strings of length (a+b+c) consisting of exactly a '0's,
// b '1's, c '2's. Order: '0's first, then '1's, then '2's at each step,
// giving lex-ascending output.

function* multinomialStrings(a, b, c) {
  if (a + b + c === 0) {
    yield ''
    return
  }
  if (a > 0) {
    for (const rest of multinomialStrings(a - 1, b, c)) yield '0' + rest
  }
  if (b > 0) {
    for (const rest of multinomialStrings(a, b - 1, c)) yield '1' + rest
  }
  if (c > 0) {
    for (const rest of multinomialStrings(a, b, c - 1)) yield '2' + rest
  }
}

// ----- Structural enumerator -----
//
// Yields { key, params } for every canonical state at depth d. params holds
// the structural decomposition so materialization doesn't need to re-parse
// the key.
//
// params = { s0, s1, s2, myTricks, oppTricks, midTrick }
//   s0, s1, s2  per-suit location strings ('0' = mover, '1' = opp, '2' = led card)
//   midTrick     whether a led card is present in some suit
//
// Suit-canonical form: s0 corresponds to trump (canonical suit 0); s1, s2
// are the two non-trump suits in lex-canonical order (s1 <= s2). The
// enumerator only yields suit-canonical keys (sorting at emission time).

function* enumerateAtDepth(d) {
  if (d < 1 || d > TRICKS_PER_ROUND) return
  const seen = new Set()

  for (let myTricks = 0; myTricks <= TRICKS_PER_ROUND - d; myTricks++) {
    const oppTricks = TRICKS_PER_ROUND - d - myTricks
    if (oppTricks < 0 || oppTricks > TRICKS_PER_ROUND) continue

    for (const midTrick of [false, true]) {
      const nMe = d
      const nOpp = midTrick ? d - 1 : d
      const nLed = midTrick ? 1 : 0
      if (nOpp < 0) continue
      const totalInPlay = nMe + nOpp + nLed // = 2d

      // Enumerate suit-size triples (k0, k1, k2) summing to totalInPlay.
      const maxK0 = Math.min(MAX_TRUMP_IN_PLAY, totalInPlay)
      const maxK = Math.min(RANKS_PER_SUIT, totalInPlay)
      for (let k0 = 0; k0 <= maxK0; k0++) {
        for (let k1 = 0; k1 <= Math.min(maxK, totalInPlay - k0); k1++) {
          const k2 = totalInPlay - k0 - k1
          if (k2 < 0 || k2 > RANKS_PER_SUIT) continue

          // Enumerate mover's '0' count per suit: sum a_i = nMe.
          for (let a0 = 0; a0 <= Math.min(k0, nMe); a0++) {
            for (let a1 = 0; a1 <= Math.min(k1, nMe - a0); a1++) {
              const a2 = nMe - a0 - a1
              if (a2 < 0 || a2 > k2) continue

              // Distribute the (0 or 1) led-card slot across suits. The led
              // card occupies one of the non-'0' positions in its suit.
              const ledSuitChoices = nLed === 0 ? [-1] : [0, 1, 2]
              for (const ledSuit of ledSuitChoices) {
                const c0 = ledSuit === 0 ? 1 : 0
                const c1 = ledSuit === 1 ? 1 : 0
                const c2 = ledSuit === 2 ? 1 : 0

                const b0 = k0 - a0 - c0
                const b1 = k1 - a1 - c1
                const b2 = k2 - a2 - c2
                if (b0 < 0 || b1 < 0 || b2 < 0) continue
                if (b0 + b1 + b2 !== nOpp) continue

                // Yield every interleaving within each suit and emit the
                // suit-canonical key. The non-trump suits (1, 2) are
                // interchangeable, so we dedupe by sorted (s1, s2).
                for (const s0 of multinomialStrings(a0, b0, c0)) {
                  for (const s1raw of multinomialStrings(a1, b1, c1)) {
                    for (const s2raw of multinomialStrings(a2, b2, c2)) {
                      const [s1, s2] = s1raw <= s2raw ? [s1raw, s2raw] : [s2raw, s1raw]
                      const key = `${s0}|${s1}|${s2}|${myTricks},${oppTricks}`
                      if (seen.has(key)) continue
                      seen.add(key)
                      yield {
                        key,
                        params: { s0, s1, s2, myTricks, oppTricks, midTrick },
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ----- Materialization -----
//
// Build a representative concrete world from canonical params. Any valid
// rank-assignment that preserves the canonical structure produces a world
// whose canonKey equals the original key — that's what verify-enum checks.
//
// Conventions:
//   canonical suit 0 = 'bells' (trump)
//   canonical suit 1 = 'keys'
//   canonical suit 2 = 'moons'
//   mover = BOT
//   leader = HUMAN when midTrick, else BOT

const TRUMP_SUIT = 'bells'
const SUIT_BY_CANON = ['bells', 'keys', 'moons']

function buildSuitCards(suitName, locString, startRank, sinks) {
  for (let i = 0; i < locString.length; i++) {
    const card = {
      suit: suitName,
      rank: startRank + i,
      id: `${suitName}-${startRank + i}`,
    }
    const loc = locString[i]
    if (loc === '0') sinks.bot.push(card)
    else if (loc === '1') sinks.human.push(card)
    else if (loc === '2') {
      if (sinks.led) throw new Error('multiple led cards in materialization')
      sinks.led = card
    } else {
      throw new Error(`unknown location code: ${loc}`)
    }
  }
}

function materialize({ s0, s1, s2, myTricks, oppTricks, midTrick }) {
  if (s0.length > MAX_TRUMP_IN_PLAY) {
    throw new Error(`trump suit in-play count ${s0.length} exceeds max ${MAX_TRUMP_IN_PLAY}`)
  }
  const sinks = { bot: [], human: [], led: null }

  // In-play trump-suit ranks: 1..L0. Trump card rank: L0 + 1 (out of play).
  buildSuitCards(TRUMP_SUIT, s0, 1, sinks)
  buildSuitCards(SUIT_BY_CANON[1], s1, 1, sinks)
  buildSuitCards(SUIT_BY_CANON[2], s2, 1, sinks)

  const trumpRank = s0.length + 1
  const trump = {
    suit: TRUMP_SUIT,
    rank: trumpRank,
    id: `${TRUMP_SUIT}-${trumpRank}`,
  }

  const mover = BOT
  const leader = midTrick ? HUMAN : BOT
  const tricksRemaining = TRICKS_PER_ROUND - myTricks - oppTricks
  const trickNum = TRICKS_PER_ROUND - tricksRemaining + 1

  return {
    humanHand: sinks.human,
    botHand: sinks.bot,
    trump,
    ledCard: sinks.led,
    awaiting: mover,
    leader,
    tricksWon: { bot: myTricks, human: oppTricks },
    score: { bot: 0, human: 0 },
    roundNum: 1,
    trickNum,
    phase: 'playing',
    trickHistory: [],
    lastTrick: null,
  }
}

// ----- Subcommands -----

function cmdVerifyEnum() {
  // Verify at every depth we'll touch in the build. d=1..10 is generous;
  // earlier depths are smaller anyway. At d > ~6 the count explodes, so
  // we cap to keep the verify fast — sizing/build will run independently.
  const verifyMaxDepth = 5
  let totalChecked = 0
  let mismatches = 0
  console.log(`verify-enum: depths 1..${verifyMaxDepth}`)
  for (let d = 1; d <= verifyMaxDepth; d++) {
    let dCount = 0
    let dMismatch = 0
    for (const { key, params } of enumerateAtDepth(d)) {
      const world = materialize(params)
      const recovered = canonKey(world)
      if (recovered !== key) {
        if (dMismatch === 0) {
          console.log(`MISMATCH at d=${d}:`)
          console.log(`  enumerated:  ${key}`)
          console.log(`  canonKey gave: ${recovered}`)
          console.log(`  params: ${JSON.stringify(params)}`)
        }
        dMismatch++
      }
      dCount++
    }
    console.log(`  d=${d}: ${dCount} states, ${dMismatch} mismatches`)
    totalChecked += dCount
    mismatches += dMismatch
  }
  if (mismatches > 0) {
    console.log(`FAIL: ${mismatches} / ${totalChecked} mismatches`)
    process.exit(1)
  }
  console.log(`PASS: ${totalChecked} states verified, 0 mismatches`)
}

function cmdSize() {
  console.log('Pass 1 — sizing canonical states per depth (packed-key format)')
  // V8's Set caps around 16.7M entries; the dedup set inside the enumerator
  // at d=8 exceeds that. Sizing past d=7 isn't useful anyway — that depth
  // already projects well over the 50 MB blob cap.
  const maxDepthForSizing = 7
  const counts = []
  for (let d = 1; d <= maxDepthForSizing; d++) {
    const t0 = Date.now()
    let count = 0
    for (const _ of enumerateAtDepth(d)) count++
    const ms = Date.now() - t0
    counts.push({ depth: d, count, ms })
    console.log(`  d=${d}: ${count.toLocaleString()} states (${ms} ms)`)
  }

  console.log('')
  console.log('Cumulative blob size at each N_MAX (packed-key format):')
  let cumEntries = 0
  let recommendedNMax = 0
  for (const { depth, count } of counts) {
    cumEntries += count
    const entryWidth = entryByteWidth(depth)
    const projBytes = FILE_HEADER_BYTES + cumEntries * entryWidth
    const fits = projBytes <= TARGET_BLOB_BYTES
    if (fits) recommendedNMax = depth
    console.log(
      `  N_MAX=${depth}: ${cumEntries.toLocaleString()} entries × ` +
        `${entryWidth} B/entry = ${(projBytes / 1024 / 1024).toFixed(2)} MB ` +
        `${fits ? '' : '(over 50 MB cap)'}`
    )
  }
  console.log('')
  console.log(`Recommended N_MAX = ${recommendedNMax} (largest that fits 50 MB)`)

  if (!existsSync(dirname(SIZE_CACHE_PATH))) {
    mkdirSync(dirname(SIZE_CACHE_PATH), { recursive: true })
  }
  writeFileSync(
    SIZE_CACHE_PATH,
    JSON.stringify({ counts, recommendedNMax }, null, 2)
  )
  console.log(`Cached to ${SIZE_CACHE_PATH}`)
}

function cmdBuild() {
  if (!existsSync(SIZE_CACHE_PATH)) {
    console.log('No size cache found. Run `node scripts/build-endgame.js size` first.')
    process.exit(1)
  }
  const { recommendedNMax } = JSON.parse(readFileSync(SIZE_CACHE_PATH, 'utf8'))
  const nMax = recommendedNMax
  console.log(`Pass 2 — building DB up to N_MAX = ${nMax}`)

  for (let d = 1; d <= nMax; d++) {
    const t0 = Date.now()
    let count = 0
    for (const { params } of enumerateAtDepth(d)) {
      const world = materialize(params)
      // solve() with rootMover = world.awaiting returns value in to-move
      // player's frame, which matches what endgame.put writes (Phase A
      // doubleDummy.js handles the frame conversion at the put site).
      const tt = new Map()
      solve(world, -Infinity, Infinity, tt, world.awaiting)
      count++
      if (count % 100000 === 0) {
        console.log(`  d=${d}: ${count.toLocaleString()} states processed (${Date.now() - t0} ms)`)
      }
    }
    const ms = Date.now() - t0
    const stats = endgame.stats()
    console.log(
      `  d=${d}: ${count.toLocaleString()} states, ${ms} ms, ` +
        `DB size=${stats.size.toLocaleString()}`
    )
  }

  serializeBlob(nMax)
}

function serializeBlob(nMax) {
  // Pack each (canonKey, value) into the fixed-stride format from
  // src/engine/endgame-blob.js, sort by packed-byte order, write to disk.
  const locBytes = locationBytesForNMax(nMax)
  const kw = keyByteWidth(nMax)
  const ew = entryByteWidth(nMax)

  const map = endgame.entries() // Map.entries() iterator
  const entries = []
  for (const [canonKeyStr, value] of map) {
    const buf = new Uint8Array(ew)
    packCanonKey(canonKeyStr, buf, 0, locBytes)
    // int8 value stored as uint8 (bit-pattern); reader sign-extends.
    buf[kw] = value & 0xff
    entries.push(buf)
  }

  entries.sort((a, b) => {
    for (let i = 0; i < kw; i++) {
      if (a[i] !== b[i]) return a[i] - b[i]
    }
    return 0
  })

  const totalSize = FILE_HEADER_BYTES + entries.length * ew
  const out = new Uint8Array(totalSize)
  const dv = new DataView(out.buffer)
  dv.setUint32(0, BLOB_MAGIC, true)
  dv.setUint16(4, BLOB_VERSION, true)
  // bytes 6..7 reserved (left as 0)
  dv.setUint32(8, nMax, true)
  dv.setUint32(12, entries.length, true)

  let off = FILE_HEADER_BYTES
  for (const e of entries) {
    out.set(e, off)
    off += ew
  }

  writeFileSync(BLOB_PATH, out)
  console.log(
    `Wrote ${BLOB_PATH}: ${entries.length.toLocaleString()} entries, ` +
      `${(out.length / 1024 / 1024).toFixed(2)} MB ` +
      `(N_MAX=${nMax}, entryWidth=${ew} B)`
  )
}

// ----- Entry point -----

const cmd = process.argv[2]
switch (cmd) {
  case 'verify-enum':
    cmdVerifyEnum()
    break
  case 'size':
    cmdSize()
    break
  case 'build':
    cmdBuild()
    break
  default:
    console.log('Usage: node scripts/build-endgame.js {verify-enum|size|build}')
    process.exit(1)
}
