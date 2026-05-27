// The v1 AI engine: pick a uniform-random legal card.
//
// Contract — same shape as tic-tac-toe-ai-engine's engines:
//   bestMove(state) -> a card from state.botHand
// Caller must only invoke when state.awaiting === BOT.

import { legalMoves } from './game.js'

export function bestMove(state) {
  const legal = legalMoves(state.botHand, state.ledCard)
  return legal[Math.floor(Math.random() * legal.length)]
}
