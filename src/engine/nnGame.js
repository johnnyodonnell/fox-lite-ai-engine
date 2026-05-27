// Glue between the pure rules (game.js) and the search / network engines.
//
// Provides:
//   - the info-set view a bot legitimately sees (no peeking at humanHand)
//   - a determinization sampler: fills in the opponent's hand + unused pile
//     consistent with everything the bot has observed
//   - perfect-info world-state helpers that MCTS rollouts apply directly
//   - encode(): the network input vector (placeholder for Phase 3)
//
// The bot ALWAYS plays the JS "bot" seat, but inside an MCTS rollout the
// mover alternates — we keep things sane by storing all values in the
// bot's frame (signed margin), and flipping sign at PUCT-time when the
// human seat is to move.

import {
  SUITS,
  RANKS,
  HUMAN,
  BOT,
  TRICKS_PER_ROUND,
  cardId,
  legalMoves,
  playCard,
  advanceAfterTrick,
  scoreForTricks,
} from './game.js'

export function allCards() {
  const out = []
  for (const s of SUITS) for (const r of RANKS) out.push({ suit: s, rank: r, id: cardId(s, r) })
  return out
}

// What the bot legitimately sees. We deliberately drop humanHand here so the
// search code can't accidentally cheat.
export function botInfoset(state) {
  return {
    botHand: state.botHand,
    trump: state.trump,
    trickHistory: state.trickHistory,
    leader: state.leader,
    ledCard: state.ledCard,
    awaiting: state.awaiting,
    tricksWon: state.tricksWon,
    score: state.score,
    roundNum: state.roundNum,
    trickNum: state.trickNum,
    phase: state.phase,
  }
}

// Suits the opponent has been observed to be void in. Inferred from any
// completed trick where the opponent failed to follow the led suit.
export function opponentVoidSuits(infoset) {
  const voids = new Set()
  const byTrick = new Map()
  for (const ev of infoset.trickHistory) {
    if (!byTrick.has(ev.trick)) byTrick.set(ev.trick, [])
    byTrick.get(ev.trick).push(ev)
  }
  for (const events of byTrick.values()) {
    if (events.length < 2) continue // current trick mid-play
    const [lead, follow] = events
    if (follow.player === HUMAN && follow.card.suit !== lead.card.suit) {
      voids.add(lead.card.suit)
    }
  }
  return voids
}

// Sample a world consistent with the bot's information set: an opponent hand
// (size = 13 minus opponent's plays so far) and the unused pile, drawn from
// the cards the bot hasn't seen and respecting any inferred suit voids.
export function sampleDeterminization(infoset, rng = Math.random) {
  const seen = new Set()
  for (const c of infoset.botHand) seen.add(c.id)
  seen.add(infoset.trump.id)
  for (const ev of infoset.trickHistory) seen.add(ev.card.id)

  const unseen = allCards().filter((c) => !seen.has(c.id))
  const voids = opponentVoidSuits(infoset)
  const allowedForOpp = unseen.filter((c) => !voids.has(c.suit))
  const mustGoToUnused = unseen.filter((c) => voids.has(c.suit))

  const opponentPlayed = infoset.trickHistory.filter((ev) => ev.player === HUMAN).length
  const opponentHandSize = 13 - opponentPlayed

  if (allowedForOpp.length < opponentHandSize) {
    throw new Error(
      `Determinization impossible: ${allowedForOpp.length} non-void cards ` +
      `available, opponent must hold ${opponentHandSize}`
    )
  }

  // Fisher-Yates on the candidate pool; first `opponentHandSize` cards form
  // the opponent's hand, the rest plus all void-suit cards form the unused pile.
  const pool = allowedForOpp.slice()
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[pool[i], pool[j]] = [pool[j], pool[i]]
  }
  const opponentHand = pool.slice(0, opponentHandSize)
  const unusedPile = [...pool.slice(opponentHandSize), ...mustGoToUnused]
  return { opponentHand, unusedPile }
}

// Build a perfect-info world state from the bot's view + a sampled
// determinization. The result is shape-compatible with the game.js state, so
// playCard / advanceAfterTrick apply directly inside MCTS rollouts.
export function worldFromDeterminization(state, det) {
  return { ...state, humanHand: det.opponentHand }
}

// One half-move of the world. After a follower plays we auto-advance past
// the trick-complete phase so MCTS sees seamless transitions.
export function stepWorld(world, card) {
  let next = playCard(world, card)
  while (next.phase === 'trick-complete') next = advanceAfterTrick(next)
  return next
}

export function isWorldTerminal(world) {
  return world.phase === 'round-over' || world.phase === 'match-over'
}

// Signed margin / 6, always from the bot's frame. Range: [-1, 1].
export function botFrameValue(world) {
  const botPts = scoreForTricks(world.tricksWon.bot)
  const humanPts = scoreForTricks(world.tricksWon.human)
  return (botPts - humanPts) / 6
}

// Uniform-random rollout to round-end. Used by Phase 2's search-only engine
// as the value estimator at non-terminal MCTS leaves.
export function rolloutValue(world, rng = Math.random) {
  let s = world
  while (!isWorldTerminal(s)) {
    const handKey = s.awaiting === HUMAN ? 'humanHand' : 'botHand'
    const legal = legalMoves(s[handKey], s.ledCard)
    const card = legal[Math.floor(rng() * legal.length)]
    s = stepWorld(s, card)
  }
  return botFrameValue(s)
}

// --- network input encoding (Phase 3 will consume this) -------------------
//
// Layout, in order:
//   own hand                    33   (mover's hand, one-hot per card)
//   played pile                 33   (all cards already played this round)
//   trump suit                   3   (one-hot)
//   trump card                  33   (one-hot — specific identity matters
//                                     because the trump card itself sits in
//                                     someone's hand / the unused pile)
//   led card (or none)          34   (33 one-hot + 1 "no led card" flag)
//   self tricks-won this round  14   (one-hot 0..13)
//   opp tricks-won this round   14
//   opp suit-void flags          3   (observed voids only)
//   "I am leader of this trick"  1
//   trick number                13   (one-hot 1..13)
//   self match score             1   (scalar / TARGET_SCORE)
//   opp match score              1
//   ---
//   total                      183

const CARD_INDEX = (() => {
  const map = new Map()
  let i = 0
  for (const s of SUITS) for (const r of RANKS) map.set(cardId(s, r), i++)
  return map
})()
const NUM_CARDS = SUITS.length * RANKS.length // 33
const TARGET_SCORE = 21

export const INPUT_SIZE = 183

function setOneHot(arr, base, idx) {
  arr[base + idx] = 1
}

// Encode the moving player's information set. `mover` defaults to state.awaiting.
// During the bot's turn at the root that's BOT; inside an MCTS rollout it
// alternates as turns alternate.
export function encode(state, mover = state.awaiting) {
  const out = new Array(INPUT_SIZE).fill(0)
  let cursor = 0

  const moverIsHuman = mover === HUMAN
  const ownHand = moverIsHuman ? state.humanHand : state.botHand
  const ownTricks = moverIsHuman ? state.tricksWon.human : state.tricksWon.bot
  const oppTricks = moverIsHuman ? state.tricksWon.bot : state.tricksWon.human
  const ownScore = moverIsHuman ? state.score.human : state.score.bot
  const oppScore = moverIsHuman ? state.score.bot : state.score.human

  // own hand
  for (const c of ownHand) setOneHot(out, cursor, CARD_INDEX.get(c.id))
  cursor += NUM_CARDS

  // played pile
  for (const ev of state.trickHistory) setOneHot(out, cursor, CARD_INDEX.get(ev.card.id))
  cursor += NUM_CARDS

  // trump suit
  setOneHot(out, cursor, SUITS.indexOf(state.trump.suit))
  cursor += SUITS.length

  // trump card identity
  setOneHot(out, cursor, CARD_INDEX.get(state.trump.id))
  cursor += NUM_CARDS

  // led card
  if (state.ledCard) {
    setOneHot(out, cursor, CARD_INDEX.get(state.ledCard.id))
  } else {
    out[cursor + NUM_CARDS] = 1 // "no led card" flag
  }
  cursor += NUM_CARDS + 1

  // self tricks
  setOneHot(out, cursor, ownTricks)
  cursor += TRICKS_PER_ROUND + 1
  // opp tricks
  setOneHot(out, cursor, oppTricks)
  cursor += TRICKS_PER_ROUND + 1

  // opponent's observed voids — only meaningful when mover is the bot
  // (during a determinized rollout the "opponent" is whoever isn't moving,
  // and that side's hand is already fully visible to the search).
  if (!moverIsHuman) {
    const voids = opponentVoidSuits({ trickHistory: state.trickHistory })
    for (const s of voids) setOneHot(out, cursor, SUITS.indexOf(s))
  }
  cursor += SUITS.length

  // leader-of-this-trick flag
  out[cursor++] = state.leader === mover && state.ledCard === null ? 1 : 0

  // trick number 1..13
  setOneHot(out, cursor, state.trickNum - 1)
  cursor += TRICKS_PER_ROUND

  // scores as scalars in [0, ~1]
  out[cursor++] = ownScore / TARGET_SCORE
  out[cursor++] = oppScore / TARGET_SCORE

  if (cursor !== INPUT_SIZE) {
    throw new Error(`encoder cursor mismatch: ${cursor} != ${INPUT_SIZE}`)
  }
  return out
}
