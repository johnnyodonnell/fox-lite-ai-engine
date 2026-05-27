// PIMC (Perfect Information Monte Carlo) search.
//
// At decision time:
//   1. Sample K determinizations consistent with the bot's information set.
//   2. In each determinization, run a PUCT MCTS tree (perfect info now).
//   3. Sum visit counts at the root across all K trees.
//   4. Play the action with the most total visits.
//
// Values are stored in the bot's frame (signed margin/6, in [-1, 1]). PUCT
// score uses Q from the *mover's* frame, so we flip sign when the mover at
// a node is the human. Trick-taking doesn't strictly alternate movers (the
// trick winner leads the next trick), so we can't use the canonical
// "flip on every backup ply" pattern.

import { HUMAN, BOT, legalMoves } from './game.js'
import {
  botInfoset,
  sampleDeterminization,
  worldFromDeterminization,
  stepWorld,
  isWorldTerminal,
  botFrameValue,
  rolloutValue,
} from './nnGame.js'

const DEFAULT_C_PUCT = 1.5

function legalForMover(world) {
  const hand = world.awaiting === HUMAN ? world.humanHand : world.botHand
  return legalMoves(hand, world.ledCard)
}

class Node {
  constructor(state, prior) {
    this.state = state
    this.prior = prior
    this.visitCount = 0
    this.valueSum = 0 // bot-frame
    this.children = new Map() // cardId -> Node
    this.expanded = false
    this.isTerminal = isWorldTerminal(state)
  }

  meanValueBotFrame() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount
  }
}

function expand(node, evaluator) {
  if (node.isTerminal || node.expanded) return
  const legal = legalForMover(node.state)
  const priors = evaluator.prior(node.state, legal)
  for (let i = 0; i < legal.length; i++) {
    const child = stepWorld(node.state, legal[i])
    node.children.set(legal[i].id, new Node(child, priors[i]))
  }
  node.expanded = true
}

function leafValue(node, evaluator) {
  if (node.isTerminal) return botFrameValue(node.state)
  return evaluator.value(node.state)
}

function puctSelectChild(node, cPuct) {
  const mover = node.state.awaiting
  const moverIsBot = mover === BOT
  const sqrtParentVisits = Math.sqrt(node.visitCount)
  let bestKey = null
  let bestChild = null
  let bestScore = -Infinity
  for (const [key, child] of node.children) {
    const meanBot = child.meanValueBotFrame()
    const qForMover = moverIsBot ? meanBot : -meanBot
    const u = cPuct * child.prior * sqrtParentVisits / (1 + child.visitCount)
    const score = qForMover + u
    if (score > bestScore) {
      bestScore = score
      bestKey = key
      bestChild = child
    }
  }
  return [bestKey, bestChild]
}

// Run `numSimulations` PUCT simulations starting from `rootState`.
// `evaluator` is `{ prior(state, legalCards) -> number[], value(state) -> number }`.
// Returns the root Node with children populated.
export function runMcts(rootState, evaluator, numSimulations, cPuct = DEFAULT_C_PUCT) {
  const root = new Node(rootState, 0)
  expand(root, evaluator)

  for (let i = 0; i < numSimulations; i++) {
    let node = root
    const path = [node]
    while (node.expanded && !node.isTerminal) {
      const [, child] = puctSelectChild(node, cPuct)
      if (!child) break
      node = child
      path.push(node)
    }
    if (!node.isTerminal) expand(node, evaluator)
    const value = leafValue(node, evaluator)
    for (const n of path) {
      n.visitCount += 1
      n.valueSum += value
    }
  }
  return root
}

// Uniform-prior + rollout-value evaluator: the Phase 2 search-only engine.
export function uniformRolloutEvaluator(rng = Math.random) {
  return {
    prior(_state, legal) {
      const p = 1 / legal.length
      return legal.map(() => p)
    },
    value(state) {
      return rolloutValue(state, rng)
    },
  }
}

// Top-level PIMC orchestration. Returns the card (from state.botHand) with
// the most aggregated root-level visit count across K determinizations.
export function bestMoveByPimc(state, opts = {}) {
  const {
    numDeterminizations = 16,
    numSimulations = 64,
    cPuct = DEFAULT_C_PUCT,
    rng = Math.random,
    evaluator = uniformRolloutEvaluator(rng),
  } = opts

  // Edge case: only one legal move — skip the search entirely.
  const legal = legalMoves(state.botHand, state.ledCard)
  if (legal.length === 1) return legal[0]

  const infoset = botInfoset(state)
  const aggregated = new Map() // cardId -> total visits

  for (let d = 0; d < numDeterminizations; d++) {
    const det = sampleDeterminization(infoset, rng)
    const world = worldFromDeterminization(state, det)
    const root = runMcts(world, evaluator, numSimulations, cPuct)
    for (const [cardId, child] of root.children) {
      aggregated.set(cardId, (aggregated.get(cardId) || 0) + child.visitCount)
    }
  }

  let bestId = null
  let bestVisits = -1
  for (const [id, v] of aggregated) {
    if (v > bestVisits) {
      bestVisits = v
      bestId = id
    }
  }
  return state.botHand.find((c) => c.id === bestId) ?? legal[0]
}
