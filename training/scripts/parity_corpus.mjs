// Parity corpus (JS side): record reference outputs of every pure rule
// function in src/engine/game.js. training/scripts/parity_check.py replays the
// same inputs through training/games/foxlite.py and asserts identical output.
//
// JS is the source of truth — these rules already ship in production. The
// corpus is regenerated when JS rules change; Python is then re-verified.
//
// Coverage:
//   - scoreForTricks: full domain (n = 0..13)
//   - trickWinner   : full domain (33 led x 33 follow x 3 trump-suit)
//   - legalMoves    : hand-curated hands covering void/non-void/no-led cases
//   - playCard      : N full self-played games, every state transition logged
//
// Run (after building/installing deps): node training/scripts/parity_corpus.mjs

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  SUITS,
  RANKS,
  HUMAN,
  BOT,
  cardId,
  createGame,
  legalMoves,
  trickWinner,
  scoreForTricks,
  playCard,
  advanceAfterTrick,
  endRound,
} from '../../src/engine/game.js'
import { encode, INPUT_SIZE } from '../../src/engine/nnGame.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const OUT_PATH = path.join(here, '..', 'parity_expected.json')

const NUM_GAMES = 200            // full match traces (each plays out >= 1 round)
const MAX_MOVES_PER_MATCH = 200  // safety stop in case of a bug

function makeCard(suit, rank) {
  return { suit, rank, id: cardId(suit, rank) }
}

function allCards() {
  const cards = []
  for (const s of SUITS) for (const r of RANKS) cards.push(makeCard(s, r))
  return cards
}

// Strip the fields the parity-check compares. lastTrick is a UI-only detail;
// trickHistory is the AI-facing log and IS checked. We compare the full state
// shape but skip non-deterministic React-y references.
function snapshotState(s) {
  return {
    humanHand: s.humanHand,
    botHand: s.botHand,
    trump: s.trump,
    leader: s.leader,
    ledCard: s.ledCard,
    awaiting: s.awaiting,
    tricksWon: s.tricksWon,
    score: s.score,
    roundNum: s.roundNum,
    trickNum: s.trickNum,
    phase: s.phase,
    lastTrick: s.lastTrick,
    trickHistory: s.trickHistory,
  }
}

// --- scoreForTricks ---------------------------------------------------------
function scoreCorpus() {
  return Array.from({ length: 14 }, (_, n) => ({ n, expected: scoreForTricks(n) }))
}

// --- trickWinner -----------------------------------------------------------
function trickWinnerCorpus() {
  const cards = allCards()
  const cases = []
  for (const led of cards) {
    for (const follow of cards) {
      for (const trumpSuit of SUITS) {
        cases.push({
          led, follow, trumpSuit,
          expected: trickWinner(led, follow, trumpSuit),
        })
      }
    }
  }
  return cases
}

// --- legalMoves ------------------------------------------------------------
function legalMovesCorpus() {
  const c = (s, r) => makeCard(s, r)
  const cases = []

  // Leading (ledCard = null) — always returns the whole hand.
  cases.push({
    hand: [c('bells', 1), c('keys', 5), c('moons', 11)],
    ledCard: null,
    expected: legalMoves([c('bells', 1), c('keys', 5), c('moons', 11)], null),
  })

  // Must follow suit when able.
  const handMixed = [c('bells', 2), c('bells', 9), c('keys', 4), c('moons', 7)]
  cases.push({
    hand: handMixed,
    ledCard: c('bells', 5),
    expected: legalMoves(handMixed, c('bells', 5)),
  })

  // Void in led suit — every card is legal.
  const handVoid = [c('keys', 4), c('moons', 7), c('moons', 11)]
  cases.push({
    hand: handVoid,
    ledCard: c('bells', 5),
    expected: legalMoves(handVoid, c('bells', 5)),
  })

  // Single-card hand (last trick of a round) — that one card is legal.
  cases.push({
    hand: [c('moons', 3)],
    ledCard: c('bells', 9),
    expected: legalMoves([c('moons', 3)], c('bells', 9)),
  })

  // Lots of randomly-drawn hands across led suits to broaden coverage.
  const cards = allCards()
  function shuffle(a, rng) {
    const x = a.slice()
    for (let i = x.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[x[i], x[j]] = [x[j], x[i]]
    }
    return x
  }
  // Deterministic-ish PRNG for repeatable corpus generation.
  let seed = 1
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 0x100000000
  }
  for (let i = 0; i < 80; i++) {
    const handSize = 1 + Math.floor(rng() * 13)
    const hand = shuffle(cards, rng).slice(0, handSize)
    const led = rng() < 0.2 ? null : shuffle(cards, rng)[0]
    cases.push({
      hand, ledCard: led,
      expected: legalMoves(hand, led),
    })
  }

  return cases
}

// --- playCard / full game traces ------------------------------------------
function playCardCorpus() {
  const games = []

  for (let g = 0; g < NUM_GAMES; g++) {
    const trace = []
    let state = createGame()
    const initialDeal = {
      humanHand: state.humanHand,
      botHand: state.botHand,
      trump: state.trump,
      leader: state.leader,
    }

    let move = 0
    while (state.phase !== 'match-over' && move < MAX_MOVES_PER_MATCH) {
      if (state.phase === 'trick-complete') {
        state = advanceAfterTrick(state)
        trace.push({ kind: 'advance', after: snapshotState(state) })
        continue
      }
      if (state.phase === 'round-over') {
        state = endRound(state)
        trace.push({ kind: 'endRound', after: snapshotState(state) })
        continue
      }
      // phase === 'playing'
      const mover = state.awaiting
      const handKey = mover === HUMAN ? 'humanHand' : 'botHand'
      const legal = legalMoves(state[handKey], state.ledCard)
      // Pick deterministically — 'first legal' is fine for parity purposes
      // and lets Python re-derive each move without sharing an RNG with JS.
      const card = legal[0]
      state = playCard(state, card)
      trace.push({ kind: 'play', by: mover, card, after: snapshotState(state) })
      move++
    }

    games.push({ initialDeal, trace })
  }
  return games
}

// --- encode -----------------------------------------------------------------
// Sample real game states from the playCard traces and encode each with both
// mover=HUMAN and mover=BOT. Python's encoder must produce byte-identical
// (float64) vectors.
function encodeCorpus(gameTraces) {
  const cases = []
  const stride = 7 // sample every Nth event
  let counter = 0
  for (const g of gameTraces) {
    for (const ev of g.trace) {
      if (ev.kind !== 'play') continue
      counter++
      if (counter % stride !== 0) continue
      if (ev.after.phase !== 'playing') continue // only states with an awaiting mover are interesting
      const state = ev.after
      for (const mover of [HUMAN, BOT]) {
        cases.push({ state, mover, expected: encode(state, mover) })
      }
      if (cases.length >= 200) break
    }
    if (cases.length >= 200) break
  }
  return cases
}

// --- write ------------------------------------------------------------------
const playCardData = playCardCorpus()
const payload = {
  meta: {
    generator: 'parity_corpus.mjs',
    suits: SUITS,
    ranks: RANKS,
    humanLabel: HUMAN,
    botLabel: BOT,
    numGames: NUM_GAMES,
    inputSize: INPUT_SIZE,
  },
  scoreForTricks: scoreCorpus(),
  trickWinner: trickWinnerCorpus(),
  legalMoves: legalMovesCorpus(),
  playCard: playCardData,
  encode: encodeCorpus(playCardData),
}

fs.writeFileSync(OUT_PATH, JSON.stringify(payload))
console.log(
  `wrote parity corpus -> ${OUT_PATH}\n` +
  `  scoreForTricks  : ${payload.scoreForTricks.length} cases\n` +
  `  trickWinner     : ${payload.trickWinner.length} cases\n` +
  `  legalMoves      : ${payload.legalMoves.length} cases\n` +
  `  playCard games  : ${payload.playCard.length}\n` +
  `  encode          : ${payload.encode.length} cases`
)
