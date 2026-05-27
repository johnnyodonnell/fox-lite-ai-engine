// Measure DDE solve latency at varying hand sizes.

import { performance } from 'node:perf_hooks'
import {
  BOT, HUMAN, SUITS, cardId,
} from '../../src/engine/game.js'
import { solve } from '../../src/engine/doubleDummy.js'

const allCards = []
for (const s of SUITS) for (let r = 1; r <= 11; r++) allCards.push({ suit: s, rank: r, id: cardId(s, r) })

function world(humanHand, botHand, trump, tricksWonBot, tricksWonHuman) {
  return {
    humanHand, botHand, trump,
    leader: BOT, ledCard: null, awaiting: BOT,
    tricksWon: { human: tricksWonHuman, bot: tricksWonBot },
    score: { human: 0, bot: 0 },
    roundNum: 1, trickNum: tricksWonBot + tricksWonHuman + 1,
    phase: 'playing', lastTrick: null, trickHistory: [],
  }
}

let seed = 7
const rng = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 0x100000000 }
function shuffleInPlace(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
}

// Hand size N → 13-N completed tricks split 50/50 (or close).
for (const N of [6, 7, 8, 9, 10, 11, 12, 13]) {
  const completed = 13 - N
  const tBot = Math.floor(completed / 2)
  const tHum = completed - tBot
  const deck = allCards.slice(); shuffleInPlace(deck)
  const trump = deck[0]
  const bot = deck.slice(1, 1 + N)
  const human = deck.slice(1 + N, 1 + 2 * N)
  const w = world(human, bot, trump, tBot, tHum)
  const t0 = performance.now()
  const TIMEOUT_MS = 30000
  let result, dt
  try {
    // Time the solve; if it exceeds budget, skip the larger sizes.
    const start = performance.now()
    result = solve(w, -Infinity, +Infinity, new Map(), BOT)
    dt = performance.now() - start
  } catch (e) {
    console.log(`  N=${N}  threw: ${e.message}`)
    break
  }
  console.log(`  N=${N}  ${dt.toFixed(1).padStart(10)} ms  value=${result.value}  best=${result.bestMove}`)
  if (dt > TIMEOUT_MS) {
    console.log(`  (exceeds ${TIMEOUT_MS}ms — stopping)`)
    break
  }
}
