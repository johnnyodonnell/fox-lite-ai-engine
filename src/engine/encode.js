// Canonical NN input encoding for the Fox-Lite self-play engine.
//
// Mover-frame ("self" = the player to move) and SUIT-CANONICALIZED: suits are
// permuted so the trump suit is always canonical slot 0, and the two non-trump
// suits map to slots 1 and 2 in ascending real-suit order. The policy head's
// 33 outputs are over canonical card slots; map back with realCardFromCanonIndex.
//
// This is the single source of truth the Rust (foxlite_core) and Python
// (training/encode.py) encoders are parity-checked against.

import { SUITS, RANKS, cardId, TRICKS_PER_ROUND, TARGET_SCORE, HUMAN } from './game.js'

export const NUM_SUITS = SUITS.length // 3
export const NUM_RANKS = RANKS.length // 11
export const NUM_CARDS = NUM_SUITS * NUM_RANKS // 33

// Layout (canonical): a history-token block followed by one-hot static blocks.
//
// History tokens (v3): one token per COMPLETED trick of the current round, in
// DESCENDING order (slot 0 = most recent completed trick). The in-progress
// trick's lead is NOT a token — it lives in the static led-card block. Each
// token is [lead card index, follow card index, led-by-self, valid]. Padded
// slots are all-zero; the valid bit disambiguates padding from a real card
// index 0 and is the net's attention/readout mask.
export const HIST_TOKENS = TRICKS_PER_ROUND - 1 // 12 (max completed tricks at a decision)
export const TOKEN_FEATS = 4 // [lead card 0..32, follow card 0..32, led-by-self 0/1, valid 0/1]
const HIST = HIST_TOKENS * TOKEN_FEATS // 48
const OWN_HAND = NUM_CARDS // 33
const TRUMP_RANK = NUM_RANKS // 11
const LED = NUM_CARDS + 1 // 34 (led card one-hot + "no led / I'm leading" slot)
const SELF_TRICKS = TRICKS_PER_ROUND + 1 // 14 (0..13)
const OPP_TRICKS = TRICKS_PER_ROUND + 1 // 14
const TRICK_NUM = TRICKS_PER_ROUND // 13 (1..13)
const SCORE_SLOTS = TARGET_SCORE // 21 (0..20)

export const INPUT_SIZE =
  HIST +
  OWN_HAND +
  TRUMP_RANK +
  LED +
  SELF_TRICKS +
  OPP_TRICKS +
  TRICK_NUM +
  SCORE_SLOTS +
  SCORE_SLOTS // 209

const suitIndex = (suit) => SUITS.indexOf(suit)

// Map a real suit index to its canonical slot given the trump suit index.
export function canonSuit(realSuitIdx, trumpIdx) {
  if (realSuitIdx === trumpIdx) return 0
  let slot = 1
  for (let s = 0; s < NUM_SUITS; s++) {
    if (s !== trumpIdx && s < realSuitIdx) slot++
  }
  return slot
}

function realSuitFromCanon(canonSlot, trumpIdx) {
  if (canonSlot === 0) return trumpIdx
  const nonTrump = []
  for (let s = 0; s < NUM_SUITS; s++) if (s !== trumpIdx) nonTrump.push(s)
  return nonTrump[canonSlot - 1]
}

// Canonical card index (0..32) for a card object {suit, rank}.
export function canonCardIndex(card, trumpIdx) {
  return canonSuit(suitIndex(card.suit), trumpIdx) * NUM_RANKS + (card.rank - 1)
}

// Inverse: canonical card slot -> a real card {suit, rank, id}.
export function realCardFromCanonIndex(ci, trumpIdx) {
  const canonSlot = Math.floor(ci / NUM_RANKS)
  const rank = (ci % NUM_RANKS) + 1
  const realSuitIdx = realSuitFromCanon(canonSlot, trumpIdx)
  const suit = SUITS[realSuitIdx]
  return { suit, rank, id: cardId(suit, rank) }
}

// Encode `state` from `mover`'s perspective. Returns Float32Array(INPUT_SIZE).
export function encode(state, mover = state.awaiting) {
  const out = new Float32Array(INPUT_SIZE)
  const trumpIdx = suitIndex(state.trump.suit)
  const moverIsHuman = mover === HUMAN

  const ownHand = moverIsHuman ? state.humanHand : state.botHand
  const selfTricks = moverIsHuman ? state.tricksWon.human : state.tricksWon.bot
  const oppTricks = moverIsHuman ? state.tricksWon.bot : state.tricksWon.human
  const selfScore = moverIsHuman ? state.score.human : state.score.bot
  const oppScore = moverIsHuman ? state.score.bot : state.score.human

  let cur = 0
  // history tokens: completed tricks, most recent first. Events arrive in play
  // order as (lead, follow) pairs; a trailing single event is the in-progress
  // trick's lead and is skipped here (it equals state.ledCard).
  const events = state.trickHistory
  const nComplete = Math.floor(events.length / 2)
  if (nComplete > HIST_TOKENS) {
    throw new Error(`completed tricks ${nComplete} > ${HIST_TOKENS}`)
  }
  for (let t = 0; t < nComplete; t++) {
    const lead = events[2 * (nComplete - 1 - t)]
    const follow = events[2 * (nComplete - 1 - t) + 1]
    const base = cur + t * TOKEN_FEATS
    out[base] = canonCardIndex(lead.card, trumpIdx)
    out[base + 1] = canonCardIndex(follow.card, trumpIdx)
    out[base + 2] = lead.player === mover ? 1 : 0
    out[base + 3] = 1
  }
  cur += HIST
  // own hand
  for (const c of ownHand) out[cur + canonCardIndex(c, trumpIdx)] = 1
  cur += OWN_HAND
  // trump rank (suit is implied = canonical slot 0)
  out[cur + (state.trump.rank - 1)] = 1
  cur += TRUMP_RANK
  // led card + "no led / I'm leading" flag
  if (state.ledCard) out[cur + canonCardIndex(state.ledCard, trumpIdx)] = 1
  else out[cur + NUM_CARDS] = 1
  cur += LED
  // tricks won
  out[cur + Math.min(selfTricks, TRICKS_PER_ROUND)] = 1
  cur += SELF_TRICKS
  out[cur + Math.min(oppTricks, TRICKS_PER_ROUND)] = 1
  cur += OPP_TRICKS
  // trick number (1..13)
  out[cur + (state.trickNum - 1)] = 1
  cur += TRICK_NUM
  // match scores (one-hot 0..20, clamped)
  out[cur + Math.min(selfScore, SCORE_SLOTS - 1)] = 1
  cur += SCORE_SLOTS
  out[cur + Math.min(oppScore, SCORE_SLOTS - 1)] = 1
  cur += SCORE_SLOTS

  if (cur !== INPUT_SIZE) throw new Error(`encode cursor ${cur} != ${INPUT_SIZE}`)
  return out
}

// Canonical legal-move mask (Float32Array(33)) for `mover`.
export function legalMask(state, mover = state.awaiting) {
  const out = new Float32Array(NUM_CARDS)
  const trumpIdx = suitIndex(state.trump.suit)
  const hand = mover === HUMAN ? state.humanHand : state.botHand
  const led = state.ledCard
  let legal = hand
  if (led) {
    const same = hand.filter((c) => c.suit === led.suit)
    if (same.length > 0) legal = same
  }
  for (const c of legal) out[canonCardIndex(c, trumpIdx)] = 1
  return out
}
