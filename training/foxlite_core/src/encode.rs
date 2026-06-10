//! Canonical NN input encoding (suit-canonicalized, mover-frame).
//!
//! Byte-for-byte parity with `src/engine/encode.js` and `training/encode.py`.
//! Suits are permuted so trump is always canonical slot 0; policy outputs are
//! over canonical card slots (invert with [`real_card_from_canon_index`]).

use crate::{Card, Player, State, NUM_CARDS, NUM_RANKS, NUM_SUITS, TARGET_SCORE, TRICKS_PER_ROUND};

// Block sizes (canonical layout).
const OWN_HAND: usize = NUM_CARDS; // 33
const PLAYED_SELF: usize = NUM_CARDS; // 33
const PLAYED_OPP: usize = NUM_CARDS; // 33
const TRUMP_RANK: usize = NUM_RANKS; // 11
const OPP_VOIDS: usize = NUM_SUITS; // 3
const LED: usize = NUM_CARDS + 1; // 34
const SELF_TRICKS: usize = TRICKS_PER_ROUND as usize + 1; // 14
const OPP_TRICKS: usize = TRICKS_PER_ROUND as usize + 1; // 14
const TRICK_NUM: usize = TRICKS_PER_ROUND as usize; // 13
const SCORE_SLOTS: usize = TARGET_SCORE as usize; // 21

pub const INPUT_SIZE: usize = OWN_HAND
    + PLAYED_SELF
    + PLAYED_OPP
    + TRUMP_RANK
    + OPP_VOIDS
    + LED
    + SELF_TRICKS
    + OPP_TRICKS
    + TRICK_NUM
    + SCORE_SLOTS
    + SCORE_SLOTS; // 230

/// Map a real suit index to its canonical slot given the trump suit index.
#[inline]
pub fn canon_suit(real_suit: u8, trump: u8) -> usize {
    if real_suit == trump {
        return 0;
    }
    let mut slot = 1usize;
    for s in 0..NUM_SUITS as u8 {
        if s != trump && s < real_suit {
            slot += 1;
        }
    }
    slot
}

#[inline]
fn real_suit_from_canon(canon_slot: usize, trump: u8) -> u8 {
    if canon_slot == 0 {
        return trump;
    }
    let mut idx = 0;
    for s in 0..NUM_SUITS as u8 {
        if s != trump {
            idx += 1;
            if idx == canon_slot {
                return s;
            }
        }
    }
    unreachable!("bad canon slot {canon_slot}")
}

/// Canonical card index (0..33) for a card given the trump suit.
#[inline]
pub fn canon_card_index(card: Card, trump: u8) -> usize {
    canon_suit(card.suit, trump) * NUM_RANKS + (card.rank as usize - 1)
}

/// Inverse: canonical card slot -> real card.
#[inline]
pub fn real_card_from_canon_index(ci: usize, trump: u8) -> Card {
    let canon_slot = ci / NUM_RANKS;
    let rank = (ci % NUM_RANKS) as u8 + 1;
    Card::new(real_suit_from_canon(canon_slot, trump), rank)
}

/// Opponent (the seat that is NOT `mover`) void suits inferred from history.
pub fn opponent_voids(state: &State, opponent: Player) -> [bool; NUM_SUITS] {
    let mut voids = [false; NUM_SUITS];
    let hist = &state.trick_history;
    let mut i = 0;
    // Events are appended in play order; each trick is a (lead, follow) pair,
    // but a trick may be incomplete at the tail (single lead event).
    while i < hist.len() {
        // gather events sharing this trick number
        let trick = hist[i].trick;
        let lead = hist[i];
        let mut j = i + 1;
        while j < hist.len() && hist[j].trick == trick {
            j += 1;
        }
        if j - i >= 2 {
            let follow = hist[i + 1];
            if follow.player == opponent && follow.card.suit != lead.card.suit {
                voids[lead.card.suit as usize] = true;
            }
        }
        i = j;
    }
    voids
}

/// Encode `state` from `mover`'s perspective into a length-`INPUT_SIZE` vector.
pub fn encode(state: &State, mover: Player) -> Vec<f32> {
    let mut out = Vec::new();
    encode_into(state, mover, &mut out);
    out
}

/// [`encode`] into a caller-owned buffer (cleared and re-zeroed), reusing its
/// capacity — the self-play pipeline encodes one leaf per simulation, so a fresh
/// per-leaf Vec was a top malloc-churn source.
pub fn encode_into(state: &State, mover: Player, out: &mut Vec<f32>) {
    out.clear();
    out.resize(INPUT_SIZE, 0.0);
    let trump = state.trump.suit;
    let opp = mover.other();

    let own_hand = state.hand(mover);
    let self_tricks = state.tricks_won[mover.idx()] as usize;
    let opp_tricks = state.tricks_won[opp.idx()] as usize;
    let self_score = state.score[mover.idx()] as usize;
    let opp_score = state.score[opp.idx()] as usize;

    let mut cur = 0;
    // own hand
    for c in own_hand {
        out[cur + canon_card_index(*c, trump)] = 1.0;
    }
    cur += OWN_HAND;
    // played by self / opp
    let played_self_base = cur;
    let played_opp_base = cur + PLAYED_SELF;
    for ev in &state.trick_history {
        let base = if ev.player == mover {
            played_self_base
        } else {
            played_opp_base
        };
        out[base + canon_card_index(ev.card, trump)] = 1.0;
    }
    cur += PLAYED_SELF + PLAYED_OPP;
    // trump rank (suit implied = canonical slot 0)
    out[cur + (state.trump.rank as usize - 1)] = 1.0;
    cur += TRUMP_RANK;
    // opponent voids
    let voids = opponent_voids(state, opp);
    for (real_suit, &is_void) in voids.iter().enumerate() {
        if is_void {
            out[cur + canon_suit(real_suit as u8, trump)] = 1.0;
        }
    }
    cur += OPP_VOIDS;
    // led card + "no led / I'm leading" flag
    match state.led_card {
        Some(led) => out[cur + canon_card_index(led, trump)] = 1.0,
        None => out[cur + NUM_CARDS] = 1.0,
    }
    cur += LED;
    // tricks won
    out[cur + self_tricks.min(TRICKS_PER_ROUND as usize)] = 1.0;
    cur += SELF_TRICKS;
    out[cur + opp_tricks.min(TRICKS_PER_ROUND as usize)] = 1.0;
    cur += OPP_TRICKS;
    // trick number (1..13)
    out[cur + (state.trick_num as usize - 1)] = 1.0;
    cur += TRICK_NUM;
    // match scores (one-hot 0..20, clamped)
    out[cur + self_score.min(SCORE_SLOTS - 1)] = 1.0;
    cur += SCORE_SLOTS;
    out[cur + opp_score.min(SCORE_SLOTS - 1)] = 1.0;
    cur += SCORE_SLOTS;

    debug_assert_eq!(cur, INPUT_SIZE);
}

/// Canonical legal-move mask (length `NUM_CARDS`) for `mover`.
pub fn legal_mask(state: &State, mover: Player) -> [f32; NUM_CARDS] {
    let mut out = [0.0f32; NUM_CARDS];
    let trump = state.trump.suit;
    let legal = crate::legal_moves(state.hand(mover), state.led_card);
    for c in legal {
        out[canon_card_index(c, trump)] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_size_is_230() {
        assert_eq!(INPUT_SIZE, 230);
    }

    #[test]
    fn canon_suit_roundtrip() {
        for trump in 0..NUM_SUITS as u8 {
            // trump maps to slot 0
            assert_eq!(canon_suit(trump, trump), 0);
            // canonical slots 0..3 invert back to distinct real suits
            let mut seen = [false; NUM_SUITS];
            for slot in 0..NUM_SUITS {
                let real = real_suit_from_canon(slot, trump);
                assert_eq!(canon_suit(real, trump), slot);
                seen[real as usize] = true;
            }
            assert!(seen.iter().all(|&b| b));
        }
    }

    #[test]
    fn canon_card_index_roundtrip() {
        for trump in 0..NUM_SUITS as u8 {
            for i in 0..NUM_CARDS {
                let card = Card::from_index(i);
                let ci = canon_card_index(card, trump);
                assert_eq!(real_card_from_canon_index(ci, trump), card);
            }
        }
    }
}
