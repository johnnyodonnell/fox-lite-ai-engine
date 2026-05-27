// The shipping AI engine — drop-in replacement for random.js.
//
//   bestMove(state) -> a card from state.botHand
//
// CURRENT (Phase 2 / 3): PIMC search with a uniform policy prior + uniform-
// random rollouts to round-end as the leaf value estimate. The full training
// pipeline (training/alphazero/) is in place and verified end-to-end with
// JS<->Python forward-pass parity to ~1e-14, but the initial 20-iteration
// smoke training did not yet beat this uniform-prior baseline at the same
// search budget — the value head's noise dominated the search. Phase 4
// scales training until net+search clears the gate, at which point this
// module switches to use nn.js + weights.json (the wiring is sketched in git
// history; revert to that pattern after re-export).

import { bestMoveByPimc } from './pimc.js'

export function bestMove(state) {
  return bestMoveByPimc(state)
}
