// End-to-end check of the browser inference path WITHOUT a browser: run the
// real exported ONNX (public/models/current.onnx) through onnxruntime (wasm) in
// Node, using the JS encoder + the same argmax/canonical-mapping as neural.js,
// and play full matches (bot=net, opponent=random). Assert every bot move is legal.
//
//   node training/scripts/test_browser_engine_e2e.mjs

import { resolve } from 'node:path'

import * as ort from 'onnxruntime-web'

import {
  createGame,
  legalMoves,
  playCard,
  advanceAfterTrick,
  endRound,
  BOT,
  SUITS,
} from '../../src/engine/game.js'
import { encode, legalMask, realCardFromCanonIndex, NUM_CARDS } from '../../src/engine/encode.js'

ort.env.wasm.wasmPaths = resolve('node_modules/onnxruntime-web/dist') + '/'
ort.env.wasm.numThreads = 1

const session = await ort.InferenceSession.create('public/models/current.onnx', {
  executionProviders: ['wasm'],
})

async function botMove(state) {
  const x = encode(state, BOT)
  const out = await session.run({ input: new ort.Tensor('float32', x, [1, x.length]) })
  const policy = out.policy.data
  const mask = legalMask(state, BOT)
  let best = -1
  let bv = -Infinity
  for (let i = 0; i < NUM_CARDS; i++) {
    if (mask[i] !== 0 && policy[i] > bv) {
      bv = policy[i]
      best = i
    }
  }
  const trumpIdx = SUITS.indexOf(state.trump.suit)
  const target = realCardFromCanonIndex(best, trumpIdx)
  return state.botHand.find((c) => c.id === target.id)
}

const MATCHES = 3
let botMoves = 0
let illegal = 0
let botMatchWins = 0
for (let m = 0; m < MATCHES; m++) {
  let s = createGame()
  while (s.phase !== 'match-over') {
    if (s.phase === 'round-over') {
      s = endRound(s)
      continue
    }
    if (s.phase === 'trick-complete') {
      s = advanceAfterTrick(s)
      continue
    }
    if (s.awaiting === BOT) {
      const card = await botMove(s)
      const legal = legalMoves(s.botHand, s.ledCard)
      if (!card || !legal.some((c) => c.id === card.id)) illegal++
      botMoves++
      s = playCard(s, card)
    } else {
      const legal = legalMoves(s.humanHand, s.ledCard)
      s = playCard(s, legal[Math.floor(Math.random() * legal.length)])
    }
  }
  if (s.score.bot >= 21 && s.score.bot >= s.score.human) botMatchWins++
}

console.log(`bot moves=${botMoves} illegal=${illegal} bot match wins=${botMatchWins}/${MATCHES}`)
if (illegal > 0) {
  console.log('BROWSER-ENGINE E2E FAILED')
  process.exit(1)
}
console.log('BROWSER-ENGINE E2E OK')
