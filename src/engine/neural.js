// The shipping AI engine — drop-in replacement for random.js.
//
//   bestMove(state) -> a card from state.botHand
//
// Phase 2: PIMC search with a uniform policy prior + uniform-random rollouts
// to round-end as the leaf value estimate. No neural network yet — but the
// search infrastructure is the one we'll keep, and Phase 3 swaps in a learned
// prior + value via src/engine/nn.js without changing this file's contract.

import { bestMoveByPimc } from './pimc.js'

export function bestMove(state) {
  return bestMoveByPimc(state)
}
