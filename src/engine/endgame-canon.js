// Canonicalization for the endgame database.
//
// Maps a world state to a string key such that two worlds with the same key
// have the same double-dummy minimax value (in the to-move player's frame).
// Both the Phase A memoization-backed DB and the Phase B retrograde builder
// must agree on this function byte-for-byte — a bug here returns wrong
// values silently.
//
// Three symmetries shrink the key-space:
//
//   1) Frame neutralization. The key is built from the to-move player's
//      perspective ("me" vs "opp"). Seat identity (bot vs human) does not
//      enter the key; values are stored in the to-move player's frame and
//      callers negate as needed.
//
//   2) Suit canonicalization. The trump suit always maps to "suit 0"; the
//      two non-trump suits are interchangeable, so their per-suit encodings
//      are sorted lexicographically. The trump card's *rank* is irrelevant
//      because it never enters play in Lite (it's a revealed marker only) —
//      only the trump suit identity matters for trickWinner.
//
//   3) Equivalent-rank canonicalization. Within each suit, only the ordinal
//      positions of in-play cards matter (not absolute ranks). The trump
//      card's rank shows up as an invisible gap in the trump suit; gaps left
//      by previously played cards are likewise invisible. This is the
//      dynamic version of the collapsedLegal pruning in doubleDummy.js.
//
// Location codes per in-play card:
//   '0' = held by the to-move player
//   '1' = held by the opponent
//   '2' = the led card (in flight; belongs to the leader/non-mover)

import { SUITS, HUMAN, BOT, TRICKS_PER_ROUND } from './game.js'

const LOC_ME = '0'
const LOC_OPP = '1'
const LOC_LED = '2'

export function tricksRemaining(world) {
  return TRICKS_PER_ROUND - world.tricksWon.bot - world.tricksWon.human
}

export function canonKey(world) {
  const me = world.awaiting
  const myHandKey = me === HUMAN ? 'humanHand' : 'botHand'
  const oppHandKey = me === HUMAN ? 'botHand' : 'humanHand'

  // Per-suit list of { rank, loc } for in-play cards only.
  const bySuit = new Map()
  for (const s of SUITS) bySuit.set(s, [])
  for (const c of world[myHandKey]) bySuit.get(c.suit).push({ rank: c.rank, loc: LOC_ME })
  for (const c of world[oppHandKey]) bySuit.get(c.suit).push({ rank: c.rank, loc: LOC_OPP })
  if (world.ledCard) {
    bySuit.get(world.ledCard.suit).push({ rank: world.ledCard.rank, loc: LOC_LED })
  }

  // Per-suit encoding: sort by absolute rank ascending; emit locations only.
  // Two in-play sets with the same ordinal-by-location pattern produce the
  // same string regardless of which absolute ranks appear.
  const encoded = new Map()
  for (const s of SUITS) {
    const arr = bySuit.get(s)
    arr.sort((a, b) => a.rank - b.rank)
    let str = ''
    for (const x of arr) str += x.loc
    encoded.set(s, str)
  }

  // Suit canonicalization: trump first, others sorted.
  const trumpSuit = world.trump.suit
  const trumpEnc = encoded.get(trumpSuit)
  const nonTrumpEncs = []
  for (const s of SUITS) if (s !== trumpSuit) nonTrumpEncs.push(encoded.get(s))
  nonTrumpEncs.sort()

  // tricksWon in to-move-player's frame.
  const myTricks = world.tricksWon[me === HUMAN ? 'human' : 'bot']
  const oppTricks = world.tricksWon[me === HUMAN ? 'bot' : 'human']

  return `${trumpEnc}|${nonTrumpEncs[0]}|${nonTrumpEncs[1]}|${myTricks},${oppTricks}`
}
