// Validate the JS canonical card-index mapping used by the browser engine
// (neural.js): realCardFromCanonIndex(canonCardIndex(card)) must round-trip for
// every trump suit and every card. Mirrors the Rust canon_card_index_roundtrip.
//
//   node training/scripts/test_encode_js_roundtrip.mjs

import { SUITS, RANKS, cardId } from '../../src/engine/game.js'
import { canonCardIndex, realCardFromCanonIndex, NUM_CARDS } from '../../src/engine/encode.js'

let checked = 0
for (let trumpIdx = 0; trumpIdx < SUITS.length; trumpIdx++) {
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      const card = { suit, rank, id: cardId(suit, rank) }
      const ci = canonCardIndex(card, trumpIdx)
      if (ci < 0 || ci >= NUM_CARDS) throw new Error(`ci out of range ${ci}`)
      const back = realCardFromCanonIndex(ci, trumpIdx)
      if (back.id !== card.id) {
        throw new Error(`roundtrip fail trump=${trumpIdx} ${card.id} -> ci ${ci} -> ${back.id}`)
      }
      checked++
    }
  }
}
// trump always lands in canonical slot 0 (indices 0..10)
for (let trumpIdx = 0; trumpIdx < SUITS.length; trumpIdx++) {
  const c = { suit: SUITS[trumpIdx], rank: 7, id: cardId(SUITS[trumpIdx], 7) }
  const ci = canonCardIndex(c, trumpIdx)
  if (Math.floor(ci / RANKS.length) !== 0) throw new Error('trump not in canon slot 0')
}
console.log(`JS encode roundtrip OK over ${checked} (trump,card) pairs`)
