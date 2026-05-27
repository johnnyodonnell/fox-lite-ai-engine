// Preserved neural-net engine — the AlphaZero-style "PIMC with trained
// policy/value network" wiring from Phase 4. Not currently the shipping
// engine: it lost head-to-head to the simpler uniform-prior PIMC at the
// same search budget (the trained net's predictions weren't accurate
// enough to beat what averaged random rollouts already provided). The
// new shipping engine is the double-dummy solver in dde.js.
//
// This module is kept intact so it's trivially revivable. To use it:
//   - regenerate src/engine/weights.json via training/scripts/export_weights.py
//   - swap App.jsx's import to './engine/neuralNet.js'
//
// The training pipeline that produces weights.json lives in training/.

import weights from './weights.json' with { type: 'json' }
import { BOT } from './game.js'
import { bestMoveByPimc } from './pimc.js'
import { CARD_INDEX, encode } from './nnGame.js'
import { forward } from './nn.js'

function networkEvaluator(w) {
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
      const mover = state.awaiting
      const { value } = forward(w, encode(state, mover))
      // Network outputs value in mover's frame; PIMC stores bot-frame.
      return mover === BOT ? value : -value
    },
  }
}

const PIMC_OPTS = {
  numDeterminizations: weights.meta?.numDeterminizations ?? 8,
  numSimulations: weights.meta?.numSimulations ?? 80,
  cPuct: weights.meta?.cPuct ?? 1.5,
  evaluator: networkEvaluator(weights),
}

export function bestMove(state) {
  return bestMoveByPimc(state, PIMC_OPTS)
}
