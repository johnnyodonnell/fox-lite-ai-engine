// The neural AI engine — a drop-in async replacement for random.js.
//
//   await bestMove(state)  // -> a card from state.botHand (no search; one forward pass)
//   preload()              // optional: warm the ONNX session before the first move
//
// Pure policy: encode the bot's infoset, run the net, mask to legal moves,
// pick the argmax (canonical) action, and map it back to a real card.

import { SUITS, BOT, legalMoves } from './game.js'
import { encode, legalMask, realCardFromCanonIndex, NUM_CARDS } from './encode.js'
import { evaluate, loadModel } from './net.js'

export async function bestMove(state) {
  const mover = state.awaiting === BOT ? BOT : state.awaiting
  const hand = state.botHand
  const x = encode(state, mover)
  const { policy } = await evaluate(x)
  const mask = legalMask(state, mover)

  let best = -1
  let bestVal = -Infinity
  for (let i = 0; i < NUM_CARDS; i++) {
    if (mask[i] !== 0 && policy[i] > bestVal) {
      bestVal = policy[i]
      best = i
    }
  }

  const trumpIdx = SUITS.indexOf(state.trump.suit)
  const target = realCardFromCanonIndex(best, trumpIdx)
  const found = hand.find((c) => c.id === target.id)
  // Fallback (should never trigger): a legal card if mapping somehow misses.
  return found || legalMoves(hand, state.ledCard)[0]
}

export function preload() {
  return loadModel()
}
