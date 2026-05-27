// Hybrid engine: trained net's policy prior + rollout value at MCTS leaves.
// This matches the training-time search exactly — Path A's value head was
// trained on rollout-grounded MCTS root-Q targets, so using rollouts at
// inference too keeps inputs and targets in distribution.

import weights from '../../src/engine/weights.json' with { type: 'json' }
import { bestMoveByPimc } from '../../src/engine/pimc.js'
import { CARD_INDEX, encode, rolloutValue } from '../../src/engine/nnGame.js'
import { forward } from '../../src/engine/nn.js'

function netPolicyRolloutEvaluator(w) {
  return {
    prior(state, legal) {
      const mover = state.awaiting
      const { policyLogits } = forward(w, encode(state, mover))
      const legalLogits = legal.map((c) => policyLogits[CARD_INDEX.get(c.id)])
      const max = Math.max(...legalLogits)
      const exps = legalLogits.map((l) => Math.exp(l - max))
      const z = exps.reduce((a, b) => a + b, 0)
      return exps.map((e) => e / z)
    },
    value(state) {
      return rolloutValue(state)
    },
  }
}

const OPTS = {
  numDeterminizations: weights.meta?.numDeterminizations ?? 8,
  numSimulations: weights.meta?.numSimulations ?? 80,
  cPuct: weights.meta?.cPuct ?? 1.5,
  evaluator: netPolicyRolloutEvaluator(weights),
}

export function bestMove(state) {
  return bestMoveByPimc(state, OPTS)
}
