// A/B strength test: WITH-blob vs WITHOUT-blob DDE.
//
// Each pair of engines plays both seats of every deal (duplicate-style
// scoring) so seat / first-mover effects wash out. The blob is installed
// once and toggled per move via endgame._setBlobEnabled().
//
// Hypothesis: engine-with completes more PIMC determinizations within the
// per-decision deadline (because the blob short-circuits deep solves),
// producing a more robust belief-weighted vote and stronger play.
//
// CLI:
//   node scripts/strength-ab.js [num_deals] [deadline_ms]
//   defaults: num_deals=10, deadline_ms=800

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  HUMAN,
  BOT,
  SUITS,
  RANKS,
  TRICKS_PER_ROUND,
  cardId,
  sortHand,
  legalMoves,
  playCard,
  advanceAfterTrick,
  scoreForTricks,
} from '../src/engine/game.js'
import { bestMove } from '../src/engine/dde.js'
import * as endgame from '../src/engine/endgame.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BLOB_PATH = resolve(__dirname, '../src/engine/endgame-data.bin')

// Silence DDE's per-decision [endgame] log; it floods the test output.
{
  const origLog = console.log
  console.log = (msg, ...rest) => {
    if (typeof msg === 'string' && msg.startsWith('[endgame]')) return
    origLog(msg, ...rest)
  }
}

// Install the blob; tests toggle its use via _setBlobEnabled.
{
  const buf = readFileSync(BLOB_PATH)
  endgame._installBlobBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
}

// Seeded RNG (mulberry32) so the same deals run for both A/B trials.
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function createDeckLocal() {
  const deck = []
  for (const suit of SUITS) {
    for (const rank of RANKS) deck.push({ suit, rank, id: cardId(suit, rank) })
  }
  return deck
}

function shuffleSeeded(arr, rng) {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function generateDeal(rng) {
  const shuffled = shuffleSeeded(createDeckLocal(), rng)
  return {
    humanHand: sortHand(shuffled.slice(0, 13)),
    botHand: sortHand(shuffled.slice(13, 26)),
    trump: shuffled[26],
  }
}

// Build a starting round state with a fixed leader.
function makeState(deal, leader) {
  return {
    humanHand: deal.humanHand,
    botHand: deal.botHand,
    trump: deal.trump,
    leader,
    ledCard: null,
    awaiting: leader,
    tricksWon: { human: 0, bot: 0 },
    score: { human: 0, bot: 0 },
    roundNum: 1,
    trickNum: 1,
    phase: 'playing',
    lastTrick: null,
    trickHistory: [],
  }
}

function step(state, card) {
  let next = playCard(state, card)
  while (next.phase === 'trick-complete') next = advanceAfterTrick(next)
  return next
}

// Flip a state so that whichever seat is to move becomes "bot" from the
// engine's perspective. The returned state is identical except that seat
// labels (in hands, awaiting, leader, tricksWon, trickHistory.player) are
// swapped. Cards themselves are unchanged.
function flipState(state) {
  return {
    ...state,
    humanHand: state.botHand,
    botHand: state.humanHand,
    awaiting: state.awaiting === BOT ? HUMAN : BOT,
    leader: state.leader === BOT ? HUMAN : BOT,
    tricksWon: { human: state.tricksWon.bot, bot: state.tricksWon.human },
    trickHistory: state.trickHistory.map((ev) => ({
      ...ev,
      player: ev.player === BOT ? HUMAN : BOT,
    })),
  }
}

function pickMoveFor(state, useBlob, deadlineMs) {
  endgame._setBlobEnabled(useBlob)
  const flipped = state.awaiting === BOT ? state : flipState(state)
  return bestMove(flipped, { deadlineMs })
}

function playRound(deal, leader, humanUsesBlob, botUsesBlob, deadlineMs) {
  let state = makeState(deal, leader)
  while (state.phase === 'playing') {
    const useBlob = state.awaiting === HUMAN ? humanUsesBlob : botUsesBlob
    const card = pickMoveFor(state, useBlob, deadlineMs)
    state = step(state, card)
  }
  return state.tricksWon
}

// CLI args
const NUM_DEALS = parseInt(process.argv[2] || '10')
const DEADLINE_MS = parseInt(process.argv[3] || '800')
const SEED = 42

console.log(
  `Strength A/B: ${NUM_DEALS} deals × 2 trials, deadlineMs=${DEADLINE_MS}, seed=${SEED}`
)
console.log('')

const rng = mulberry32(SEED)

let withTricks = 0
let withoutTricks = 0
let withPoints = 0
let withoutPoints = 0
let withWins = 0
let withoutWins = 0
let draws = 0
let totalRounds = 0

const tStart = Date.now()
for (let i = 0; i < NUM_DEALS; i++) {
  const deal = generateDeal(rng)
  const dealStart = Date.now()

  // Trial A: WITH plays bot, WITHOUT plays human.
  // (Both seats use bestMove; the toggle picks which engine each call uses.)
  const a = playRound(deal, HUMAN, false, true, DEADLINE_MS)
  withTricks += a.bot
  withoutTricks += a.human
  const aWithPts = scoreForTricks(a.bot)
  const aWithoutPts = scoreForTricks(a.human)
  withPoints += aWithPts
  withoutPoints += aWithoutPts
  if (aWithPts > aWithoutPts) withWins++
  else if (aWithPts < aWithoutPts) withoutWins++
  else draws++
  totalRounds++

  // Trial B: WITHOUT plays bot, WITH plays human.
  const b = playRound(deal, HUMAN, true, false, DEADLINE_MS)
  withTricks += b.human
  withoutTricks += b.bot
  const bWithPts = scoreForTricks(b.human)
  const bWithoutPts = scoreForTricks(b.bot)
  withPoints += bWithPts
  withoutPoints += bWithoutPts
  if (bWithPts > bWithoutPts) withWins++
  else if (bWithPts < bWithoutPts) withoutWins++
  else draws++
  totalRounds++

  console.log(
    `  deal ${i + 1}: A(with=${a.bot},without=${a.human}) ` +
      `B(without=${b.bot},with=${b.human}) ` +
      `pts A=${aWithPts}/${aWithoutPts} B=${bWithoutPts}/${bWithPts} ` +
      `(${Date.now() - dealStart}ms)`
  )
}
const totalMs = Date.now() - tStart

console.log('')
console.log('=== Aggregate ===')
console.log(`Rounds played       : ${totalRounds}`)
console.log(`Total wall time     : ${(totalMs / 1000).toFixed(1)} s`)
console.log('')
console.log(`Tricks WITH         : ${withTricks} (mean/round ${(withTricks / totalRounds).toFixed(2)})`)
console.log(`Tricks WITHOUT      : ${withoutTricks} (mean/round ${(withoutTricks / totalRounds).toFixed(2)})`)
console.log(`Tricks diff (W−Wo)  : ${withTricks - withoutTricks}`)
console.log('')
console.log(`Round points WITH   : ${withPoints} (mean ${(withPoints / totalRounds).toFixed(2)})`)
console.log(`Round points WITHOUT: ${withoutPoints} (mean ${(withoutPoints / totalRounds).toFixed(2)})`)
console.log(`Points diff (W−Wo)  : ${withPoints - withoutPoints}`)
console.log('')
console.log(`Round wins WITH     : ${withWins} / ${totalRounds}`)
console.log(`Round wins WITHOUT  : ${withoutWins} / ${totalRounds}`)
console.log(`Draws               : ${draws}`)
