// Double-dummy solver — exact alpha-beta minimax over a perfect-info world.
// Used inside PIMC: each determinization is solved exactly here, and the
// outer ensemble (dde.js) averages across worlds.
//
// Key correctness pieces:
//   - TT entries carry a bound flag (EXACT / LOWER / UPPER) so cached values
//     aren't applied outside the alpha-beta window that produced them. This
//     is the classic alpha-beta + TT correctness footgun.
//   - Equivalent-card collapsing in the move generator: two cards of the
//     same suit with no still-in-play card between them are interchangeable
//     for the rest of the round, so we try only one representative. Without
//     this, trick-1 (13×13 hands) does not finish in any acceptable time.
//   - Leaf value is the signed end-of-round point margin in the *root
//     mover's* frame. Lite's non-monotonic scoring (0-3 → 6 pts, 4-6 → 1-3
//     pts, 7-9 → 6 pts, 10-13 → 0 pts) is fully captured at the leaf;
//     intra-search reasoning is over plain real numbers so alpha-beta
//     applies as usual.

import {
  HUMAN,
  BOT,
  SUITS,
  advanceAfterTrick,
  legalMoves,
  playCard,
  scoreForTricks,
} from './game.js'
import { CARD_INDEX } from './nnGame.js'

// TT entry flag values.
const EXACT = 0
const LOWER = 1 // value is a lower bound on the true value (beta-cutoff happened)
const UPPER = 2 // value is an upper bound on the true value (no move beat alpha)

// Pack a hand as a 33-bit BigInt (one bit per card index).
function packHand(hand) {
  let pack = 0n
  for (const c of hand) pack |= 1n << BigInt(CARD_INDEX.get(c.id))
  return pack
}

// Stable key for a world. Trump suit and round-leader are constant within
// a single solve, so they live in the closure rather than the key.
function ttKey(world) {
  const hk = packHand(world.humanHand)
  const bk = packHand(world.botHand)
  const led = world.ledCard ? world.ledCard.id : '-'
  return `${hk}|${bk}|${led}|${world.leader[0]}|${world.tricksWon.bot},${world.tricksWon.human}`
}

// One half-move. Auto-advances past the trick-complete pause so the solver
// only sees `playing` and `round-over` states.
function stepWorld(world, card) {
  let next = playCard(world, card)
  while (next.phase === 'trick-complete') next = advanceAfterTrick(next)
  return next
}

function isTerminal(world) {
  return world.phase === 'round-over' || world.phase === 'match-over'
}

// Signed end-of-round point margin in the root mover's frame. Range [-6,+6].
function leafMargin(world, rootMover) {
  const bot = scoreForTricks(world.tricksWon.bot)
  const hum = scoreForTricks(world.tricksWon.human)
  const botFrame = bot - hum
  return rootMover === BOT ? botFrame : -botFrame
}

// Equivalent-card collapsing. Two cards r1 < r2 in the mover's hand of the
// same suit are interchangeable for the rest of the round when no card with
// a rank strictly between them remains in any *active* location: the
// opponent's hand, or the currently-led card. Past plays and cards in the
// unused pile are out of play and don't break equivalence.
//
// Returns one representative per equivalence class — the lowest. Choice
// within a class doesn't affect the subtree value (which is what alpha-beta
// is computing), so the PIMC vote may name an arbitrary representative.
function collapsedLegal(world) {
  const moverIsHuman = world.awaiting === HUMAN
  const moverHand = moverIsHuman ? world.humanHand : world.botHand
  const oppHand = moverIsHuman ? world.botHand : world.humanHand
  const legal = legalMoves(moverHand, world.ledCard)

  // Per-suit "blockers" — ranks of cards still in active play (opp's hand
  // and the led card). Mover's own cards are excluded — they're the things
  // we're choosing among, not separators.
  const blockers = new Map()
  for (const s of SUITS) blockers.set(s, new Set())
  for (const c of oppHand) blockers.get(c.suit).add(c.rank)
  if (world.ledCard) blockers.get(world.ledCard.suit).add(world.ledCard.rank)

  const bySuit = new Map()
  for (const c of legal) {
    if (!bySuit.has(c.suit)) bySuit.set(c.suit, [])
    bySuit.get(c.suit).push(c)
  }

  const reps = []
  for (const [suit, cards] of bySuit) {
    cards.sort((a, b) => a.rank - b.rank)
    const block = blockers.get(suit)
    reps.push(cards[0])
    let prevRank = cards[0].rank
    for (let i = 1; i < cards.length; i++) {
      const c = cards[i]
      let interrupted = false
      for (let r = prevRank + 1; r < c.rank; r++) {
        if (block.has(r)) {
          interrupted = true
          break
        }
      }
      if (interrupted) reps.push(c)
      prevRank = c.rank
    }
  }
  return reps
}

// Move ordering: try the TT's best move first (most likely to produce an
// early cutoff). Within the rest, mover's representatives are emitted in
// the order returned by collapsedLegal (per-suit ascending).
function orderMoves(legal, ttBestId) {
  if (!ttBestId) return legal
  const i = legal.findIndex((c) => c.id === ttBestId)
  if (i <= 0) return legal
  const out = legal.slice()
  const [m] = out.splice(i, 1)
  out.unshift(m)
  return out
}

// Recursive alpha-beta. Maximizes when world.awaiting === rootMover,
// minimizes otherwise. The single root-mover frame avoids the non-strictly-
// alternating-mover trap that complicates trick-taking MCTS.
export function solve(world, alpha, beta, tt, rootMover) {
  if (isTerminal(world)) {
    return { value: leafMargin(world, rootMover), bestMove: null }
  }

  const origAlpha = alpha
  const origBeta = beta
  const key = ttKey(world)
  const ttHit = tt.get(key)
  if (ttHit) {
    if (ttHit.flag === EXACT) return ttHit
    if (ttHit.flag === LOWER && ttHit.value >= beta) return ttHit
    if (ttHit.flag === UPPER && ttHit.value <= alpha) return ttHit
    if (ttHit.flag === LOWER) alpha = Math.max(alpha, ttHit.value)
    else if (ttHit.flag === UPPER) beta = Math.min(beta, ttHit.value)
    if (alpha >= beta) return ttHit
  }

  const maximizing = world.awaiting === rootMover
  const moves = orderMoves(collapsedLegal(world), ttHit?.bestMove)

  let bestVal = maximizing ? -Infinity : Infinity
  let bestMoveId = null

  for (const move of moves) {
    const next = stepWorld(world, move)
    const { value } = solve(next, alpha, beta, tt, rootMover)
    if (maximizing) {
      if (value > bestVal) {
        bestVal = value
        bestMoveId = move.id
      }
      if (bestVal > alpha) alpha = bestVal
    } else {
      if (value < bestVal) {
        bestVal = value
        bestMoveId = move.id
      }
      if (bestVal < beta) beta = bestVal
    }
    if (alpha >= beta) break // cutoff
  }

  // Flag determination uses the ORIGINAL window — not the alpha/beta we
  // adjusted along the way. Wrong here = wrong cached cutoffs later.
  let flag
  if (bestVal <= origAlpha) flag = UPPER
  else if (bestVal >= origBeta) flag = LOWER
  else flag = EXACT

  const entry = { value: bestVal, flag, bestMove: bestMoveId }
  tt.set(key, entry)
  return entry
}

// Compute the exact minimax value of EVERY legal action at the root, not
// just the optimal one. PIMC aggregation needs per-action values across
// determinizations — `solve()` alone gives only the optimal-action value,
// which produces a statistically biased estimator.
//
// Cost: |legal| separate alpha-beta searches sharing one TT. Many of the
// reachable sub-states are common across first-action choices (different
// first move → same set of cards in flight later), so TT reuse keeps this
// far below |legal|× the cost of a single solve in practice.
export function solveRoot(world, tt, rootMover) {
  const moves = collapsedLegal(world)
  const out = []
  for (const move of moves) {
    const next = stepWorld(world, move)
    const { value } = solve(next, -Infinity, +Infinity, tt, rootMover)
    out.push({ cardId: move.id, value })
  }
  return out
}
