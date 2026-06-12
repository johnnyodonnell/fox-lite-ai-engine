//! Canonical NN input encoding (suit-canonicalized, mover-frame).
//!
//! Byte-for-byte parity with `src/engine/encode.js` and `training/encode.py`.
//! Suits are permuted so trump is always canonical slot 0; policy outputs are
//! over canonical card slots (invert with [`real_card_from_canon_index`]).
//!
//! v2 layout: `[ history tokens | static one-hot blocks ]`. One token per play
//! event of the current round in play order (the in-progress trick's lead is
//! just the last token); each token is [canonical card index, played-by-self,
//! valid]. Padded slots are all-zero; the valid bit disambiguates padding from
//! a real card index 0 and is the net's attention/pooling mask.

use crate::{Card, Player, State, NUM_CARDS, NUM_RANKS, NUM_SUITS, TARGET_SCORE, TRICKS_PER_ROUND};

// Block sizes (canonical layout).
pub const HIST_TOKENS: usize = 2 * TRICKS_PER_ROUND as usize; // 26 (max events in a round)
pub const TOKEN_FEATS: usize = 3; // [card index 0..32, played-by-self 0/1, valid 0/1]
pub const HIST: usize = HIST_TOKENS * TOKEN_FEATS; // 78
const OWN_HAND: usize = NUM_CARDS; // 33
const TRUMP_RANK: usize = NUM_RANKS; // 11
const SELF_TRICKS: usize = TRICKS_PER_ROUND as usize + 1; // 14
const OPP_TRICKS: usize = TRICKS_PER_ROUND as usize + 1; // 14
const TRICK_NUM: usize = TRICKS_PER_ROUND as usize; // 13
const SCORE_SLOTS: usize = TARGET_SCORE as usize; // 21

/// Size of the static (non-token) tail of the encoding.
pub const STATIC_SIZE: usize = OWN_HAND
    + TRUMP_RANK
    + SELF_TRICKS
    + OPP_TRICKS
    + TRICK_NUM
    + SCORE_SLOTS
    + SCORE_SLOTS; // 127

pub const INPUT_SIZE: usize = HIST + STATIC_SIZE; // 205

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

/// Encode `state` from `mover`'s perspective into a length-`INPUT_SIZE` vector.
pub fn encode(state: &State, mover: Player) -> Vec<f32> {
    let mut out = vec![0.0f32; INPUT_SIZE];
    let trump = state.trump.suit;
    let opp = mover.other();

    let own_hand = state.hand(mover);
    let self_tricks = state.tricks_won[mover.idx()] as usize;
    let opp_tricks = state.tricks_won[opp.idx()] as usize;
    let self_score = state.score[mover.idx()] as usize;
    let opp_score = state.score[opp.idx()] as usize;

    let mut cur = 0;
    // history tokens (play order; padded slots stay all-zero)
    let events = &state.trick_history;
    assert!(
        events.len() <= HIST_TOKENS,
        "trickHistory length {} > {HIST_TOKENS}",
        events.len()
    );
    for (i, ev) in events.iter().enumerate() {
        out[cur + i * TOKEN_FEATS] = canon_card_index(ev.card, trump) as f32;
        out[cur + i * TOKEN_FEATS + 1] = if ev.player == mover { 1.0 } else { 0.0 };
        out[cur + i * TOKEN_FEATS + 2] = 1.0;
    }
    cur += HIST;
    // own hand
    for c in own_hand {
        out[cur + canon_card_index(*c, trump)] = 1.0;
    }
    cur += OWN_HAND;
    // trump rank (suit implied = canonical slot 0)
    out[cur + (state.trump.rank as usize - 1)] = 1.0;
    cur += TRUMP_RANK;
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
    out
}

// ---------------------------------------------------------------------------
// v1 encoding (pre-history, INPUT_SIZE 230) — kept so legacy snapshots
// (run1-run3) can be evaluated against current nets. Restored verbatim from
// the pre-5f0a8aa layout: played/voids/LED one-hot blocks, no history tokens.
// ---------------------------------------------------------------------------
const PLAYED_SELF: usize = NUM_CARDS; // 33
const PLAYED_OPP: usize = NUM_CARDS; // 33
const OPP_VOIDS: usize = NUM_SUITS; // 3
const LED: usize = NUM_CARDS + 1; // 34

pub const INPUT_SIZE_V1: usize = OWN_HAND
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

/// Opponent (the seat that is NOT `mover`) void suits inferred from history.
fn opponent_voids(state: &State, opponent: Player) -> [bool; NUM_SUITS] {
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

/// v1 encode of `state` from `mover`'s perspective: length `INPUT_SIZE_V1`.
pub fn encode_v1(state: &State, mover: Player) -> Vec<f32> {
    let mut out = vec![0.0f32; INPUT_SIZE_V1];
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

    debug_assert_eq!(cur, INPUT_SIZE_V1);
    out
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
    fn input_size_is_205() {
        assert_eq!(INPUT_SIZE, 205);
        assert_eq!(STATIC_SIZE, 127);
        assert_eq!(HIST, 78);
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
