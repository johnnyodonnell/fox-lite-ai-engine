// Double-dummy engine (DDE) — PIMC orchestration over the alpha-beta
// solver in doubleDummy.js.
//
// For each of K determinizations: sample a consistent opponent hand +
// unused pile, splice into a perfect-info world, solve exactly with
// alpha-beta, record (bestMove, value). Aggregate by mean value across
// the determinizations where that card was chosen, with vote count as the
// tie-breaker. This is the standard recipe behind strong bridge / hearts
// / spades engines.

import { BOT, legalMoves } from './game.js'
import {
  botInfoset,
  sampleDeterminization,
  worldFromDeterminization,
} from './nnGame.js'
import { bestMoveByPimc, uniformRolloutEvaluator } from './pimc.js'
import { solveRoot } from './doubleDummy.js'

const DEFAULT_K = 16

// Phase 2 PIMC fallback for the deepest-tree decisions of each round.
// Alpha-beta on 13×13 hands takes ~600 ms per solve, so a K=16 ensemble at
// trick 1 would be ~10 s per decision. PIMC-with-rollouts at the same
// budget gives strong moves in ~150 ms.
const TRICK1_PIMC_OPTS = {
  numDeterminizations: 8,
  numSimulations: 80,
  cPuct: 1.5,
  evaluator: uniformRolloutEvaluator(),
}

export function bestMove(state, opts = {}) {
  const { numDeterminizations = DEFAULT_K, rng = Math.random } = opts

  const legal = legalMoves(state.botHand, state.ledCard)
  if (legal.length === 0) return null
  if (legal.length === 1) return legal[0]

  // Alpha-beta on near-full hands is dominated by worst-case behavior on
  // imbalanced trees — empirically a single trick-2 solve can take 800 ms+
  // even with collapsing + TT, so K=16 ensemble would be 10 s+ per decision.
  // Fall back to Phase 2 PIMC for the first three tricks (N=11..13 cards
  // per hand), then resume DDE at trick 4 (N≤10) where alpha-beta is fast
  // and reliable.
  if (state.trickNum <= 3) {
    return bestMoveByPimc(state, TRICK1_PIMC_OPTS)
  }

  const infoset = botInfoset(state)

  // For each legal card, accumulate value summed across all determinizations
  // (not just the ones where it was optimal). Divide by K to get the true
  // expected value of playing that card averaged over the sampled worlds.
  const sum = new Map()
  const count = new Map()

  for (let d = 0; d < numDeterminizations; d++) {
    const det = sampleDeterminization(infoset, rng)
    const world = worldFromDeterminization(state, det)
    const tt = new Map()
    const perAction = solveRoot(world, tt, BOT)
    for (const { cardId, value } of perAction) {
      sum.set(cardId, (sum.get(cardId) ?? 0) + value)
      count.set(cardId, (count.get(cardId) ?? 0) + 1)
    }
  }

  // Pick the card with the highest mean value across all determinizations.
  // Ties broken by lowest card id (matches solver's own tiebreaker).
  let best = null
  let bestMean = -Infinity
  for (const [cardId, n] of count) {
    const mean = sum.get(cardId) / n
    if (
      mean > bestMean ||
      (mean === bestMean && (best === null || cardId < best))
    ) {
      bestMean = mean
      best = cardId
    }
  }

  return state.botHand.find((c) => c.id === best) ?? legal[0]
}
