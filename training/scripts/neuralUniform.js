import { bestMoveByPimc, uniformRolloutEvaluator } from '../../src/engine/pimc.js'
const OPTS = { numDeterminizations: 8, numSimulations: 80, cPuct: 1.5, evaluator: uniformRolloutEvaluator() }
export function bestMove(state) { return bestMoveByPimc(state, OPTS) }
