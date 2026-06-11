// Dump (state, mover, encoding, legal-mask) fixtures from the reference JS
// encoder (src/engine/encode.js) at every decision point of random games, so
// the Rust (foxlite_core) and Python (training/encode.py) encoders can be
// parity-checked bit-for-bit.
//
//   node training/scripts/dump_encode_fixtures.mjs [numGames] > training/fixtures/encode_fixtures.json
//
// Encodings are all 0/1, so we store only the set (nonzero) indices.

import {
  createGame,
  legalMoves,
  playCard,
  advanceAfterTrick,
  endRound,
  HUMAN,
} from '../../src/engine/game.js'
import { encode, legalMask, INPUT_SIZE, NUM_CARDS } from '../../src/engine/encode.js'

const serCard = (c) => (c ? { suit: c.suit, rank: c.rank } : null)

function serState(s) {
  return {
    humanHand: s.humanHand.map(serCard),
    botHand: s.botHand.map(serCard),
    trump: serCard(s.trump),
    ledCard: serCard(s.ledCard),
    tricksWon: { ...s.tricksWon },
    score: { ...s.score },
    roundNum: s.roundNum,
    trickNum: s.trickNum,
    leader: s.leader,
    awaiting: s.awaiting,
    trickHistory: s.trickHistory.map((ev) => ({
      trick: ev.trick,
      player: ev.player,
      card: serCard(ev.card),
    })),
  }
}

function setIdx(arr) {
  const r = []
  for (let i = 0; i < arr.length; i++) if (arr[i] !== 0) r.push(i)
  return r
}

const numGames = parseInt(process.argv[2] || '60', 10)
const cases = []
for (let g = 0; g < numGames; g++) {
  let state = createGame()
  while (state.phase !== 'match-over') {
    if (state.phase === 'round-over') {
      state = endRound(state)
      continue
    }
    if (state.phase === 'trick-complete') {
      state = advanceAfterTrick(state)
      continue
    }
    // phase === 'playing': record a decision fixture, then play a random move
    const mover = state.awaiting
    cases.push({
      state: serState(state),
      mover,
      enc: setIdx(encode(state, mover)),
      mask: setIdx(legalMask(state, mover)),
    })
    const hand = mover === HUMAN ? state.humanHand : state.botHand
    const legal = legalMoves(hand, state.ledCard)
    state = playCard(state, legal[Math.floor(Math.random() * legal.length)])
  }
}

process.stdout.write(
  JSON.stringify({ inputSize: INPUT_SIZE, numCards: NUM_CARDS, cases })
)
