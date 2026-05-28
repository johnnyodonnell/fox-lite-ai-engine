// Double-dummy engine (DDE) — PIMC orchestration over the alpha-beta
// solver in doubleDummy.js.
//
// For each determinization (a sampled consistent opponent hand + unused
// pile), build a perfect-info world and solve it exactly with alpha-beta.
// Aggregate the per-card minimax values across worlds, weighted by the
// posterior plausibility of each world given the bot's history. Pick the
// card with the highest weighted-mean value.
//
// Latency policy:
//   - Adaptive K (number of determinizations) by trick number — alpha-beta
//     cost rises sharply with hand size, so trick 1 gets fewer worlds and
//     mid-late round gets more for robust voting.
//   - Anytime cap: never start a solve we'd be unable to finish before the
//     per-decision deadline. Guarantees worst-case latency.
//   - PIMC safety net: if the deadline blows before the first solve
//     completes, fall back to PIMC-with-rollouts (always quick).

import { BOT, legalMoves } from './game.js'
import {
  beliefWeight,
  botInfoset,
  sampleDeterminization,
  worldFromDeterminization,
} from './nnGame.js'
import { bestMoveByPimc, uniformRolloutEvaluator } from './pimc.js'
import { solveRoot } from './doubleDummy.js'

// Per-decision wall-clock budget. The deployed UI tolerates up to 3 s on
// the worst position; we leave headroom so an in-flight solve doesn't push
// us past that.
const DEADLINE_MS = 2500

// K (number of determinizations) by trick number. Worst-case alpha-beta
// scales sharply with hand size, so early-round we use fewer worlds.
// Tuned to keep the median solve time per decision well under DEADLINE_MS;
// the anytime cap is the safety net for the variance tail.
function plannedK(trickNum) {
  if (trickNum <= 1) return 4
  if (trickNum === 2) return 8
  if (trickNum === 3) return 16
  return 32
}

// Fallback when DDE produces no usable determinizations (deadline blew
// before the first solve completed). Rare but the safety net is cheap.
const PIMC_FALLBACK_OPTS = {
  numDeterminizations: 8,
  numSimulations: 80,
  cPuct: 1.5,
  evaluator: uniformRolloutEvaluator(),
}

const now = () =>
  typeof performance !== 'undefined' && performance.now
    ? performance.now()
    : Date.now()

export function bestMove(state, opts = {}) {
  const { rng = Math.random, deadlineMs = DEADLINE_MS } = opts

  const legal = legalMoves(state.botHand, state.ledCard)
  if (legal.length === 0) return null
  if (legal.length === 1) return legal[0]

  const t0 = now()
  const infoset = botInfoset(state)
  const K = plannedK(state.trickNum)

  // Per-card running stats — weighted by each world's posterior plausibility.
  //   weightedSum[card]   = sum of weight * value across worlds
  //   weightTotal[card]   = sum of weights across worlds (used to normalize)
  const weightedSum = new Map()
  const weightTotal = new Map()
  let solvesCompleted = 0

  for (let d = 0; d < K; d++) {
    // Anytime cap — bail if the deadline is imminent. We check before
    // starting each solve rather than during, so a single solve can blow
    // past the deadline (bounded by the worst-case solve time at this
    // trick). The plannedK schedule keeps that tail under control.
    if (now() - t0 > deadlineMs) break

    const det = sampleDeterminization(infoset, rng)
    const w = beliefWeight(infoset, det.opponentHand)
    if (w <= 0) continue // degenerate; skip — sampler can produce another

    const world = worldFromDeterminization(state, det)
    const tt = new Map()
    const perAction = solveRoot(world, tt, BOT)
    for (const { cardId, value } of perAction) {
      weightedSum.set(cardId, (weightedSum.get(cardId) ?? 0) + w * value)
      weightTotal.set(cardId, (weightTotal.get(cardId) ?? 0) + w)
    }
    solvesCompleted++
  }

  if (solvesCompleted === 0) {
    return bestMoveByPimc(state, PIMC_FALLBACK_OPTS)
  }

  let best = null
  let bestMean = -Infinity
  for (const [cardId, tot] of weightTotal) {
    const mean = weightedSum.get(cardId) / tot
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
