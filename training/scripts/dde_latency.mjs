// Measure per-decision latency of the DDE engine across a real game.
// Starts from a fresh deal, plays full matches (both seats using DDE),
// records the wall-clock time of each bestMove() call. Reports
// percentiles and the slowest decisions (which is trick 1 territory).

import { performance } from 'node:perf_hooks'

import {
  HUMAN, BOT, advanceAfterTrick, createGame, endRound, playCard,
} from '../../src/engine/game.js'
import { bestMove } from '../../src/engine/neural.js'

function swapPlayer(p) {
  if (p === HUMAN) return BOT
  if (p === BOT) return HUMAN
  return p
}
function swapSeats(state) {
  return {
    ...state,
    humanHand: state.botHand,
    botHand: state.humanHand,
    leader: swapPlayer(state.leader),
    awaiting: swapPlayer(state.awaiting),
    tricksWon: { human: state.tricksWon.bot, bot: state.tricksWon.human },
    score: { human: state.score.bot, bot: state.score.human },
    lastTrick: state.lastTrick && {
      ...state.lastTrick,
      leader: swapPlayer(state.lastTrick.leader),
      winner: swapPlayer(state.lastTrick.winner),
    },
    trickHistory: state.trickHistory.map((ev) => ({ ...ev, player: swapPlayer(ev.player) })),
  }
}

function playMatch(timings) {
  let state = createGame()
  while (state.phase !== 'match-over') {
    if (state.phase === 'round-over') { state = endRound(state); continue }
    if (state.phase === 'trick-complete') { state = advanceAfterTrick(state); continue }
    const seat = state.awaiting
    const view = seat === BOT ? state : swapSeats(state)
    const t0 = performance.now()
    const card = bestMove(view)
    const dt = performance.now() - t0
    timings.push({ ms: dt, trickNum: state.trickNum, ledCard: !!state.ledCard })
    state = playCard(state, card)
  }
}

const NUM_MATCHES = Number(process.argv[2] ?? '1')
const timings = []
const tStart = performance.now()
for (let i = 0; i < NUM_MATCHES; i++) {
  playMatch(timings)
  process.stdout.write(`\r  played ${i + 1}/${NUM_MATCHES}`)
}
const tTotal = performance.now() - tStart
console.log()

timings.sort((a, b) => a.ms - b.ms)
const pct = (q) => timings[Math.min(timings.length - 1, Math.floor(timings.length * q))].ms
console.log()
console.log(`  matches      ${NUM_MATCHES}`)
console.log(`  decisions    ${timings.length}`)
console.log(`  total time   ${(tTotal / 1000).toFixed(1)} s`)
console.log(`  median       ${pct(0.5).toFixed(1)} ms`)
console.log(`  p90          ${pct(0.9).toFixed(1)} ms`)
console.log(`  p99          ${pct(0.99).toFixed(1)} ms`)
console.log(`  max          ${timings[timings.length - 1].ms.toFixed(1)} ms`)
console.log()
console.log('  slowest 5:')
for (let i = timings.length - 1; i >= Math.max(0, timings.length - 5); i--) {
  const t = timings[i]
  console.log(`    ${t.ms.toFixed(1).padStart(8)} ms  trick ${t.trickNum}  ${t.ledCard ? 'following' : 'leading'}`)
}
