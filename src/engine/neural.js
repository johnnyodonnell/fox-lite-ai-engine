// The shipping AI engine — drop-in replacement for random.js.
//
//   bestMove(state) -> a card from state.botHand
//
// CURRENT: PIMC search with a uniform policy prior + uniform-random rollouts
// to round-end as the leaf value estimate. The training pipeline
// (training/alphazero/) is in place and verified end-to-end with JS<->Python
// forward-pass parity to ~1e-14, but neither the 20-iter nor 100-iter
// training runs produced a network that beats this uniform-prior baseline
// at the same search budget — the value head is degrading rather than
// improving. Once the value-target problem is diagnosed and fixed, this
// module will switch to import nn.js + weights.json.

import { bestMoveByPimc } from './pimc.js'

export function bestMove(state) {
  return bestMoveByPimc(state)
}
