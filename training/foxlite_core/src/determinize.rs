//! Determinization for ISMCTS — sample a concrete hidden world consistent with
//! the *searcher's* information set.
//!
//! The searcher (the seat we are searching for) knows: its own hand, the trump,
//! and the full public play history (hence every played card and any suit the
//! opponent has shown void in). It does NOT know the opponent's current hand or
//! which 6 cards are the unused pile. A determinization resamples those hidden
//! cards: the opponent's current hand (its public count) plus the 6 unused, drawn
//! from the unseen pool, subject to the void constraints.
//!
//! Correctness note: the true world is always a feasible assignment, and *all*
//! unseen cards of an opponent-void suit must lie in the (≤6) unused pile (the
//! opponent holds none, and the searcher's / played ones are not unseen). So the
//! forced-to-unused set never exceeds 6 and the constraint-aware deal below can
//! never fail — no rejection loop is needed.

use rand::Rng;

use crate::encode::opponent_voids;
use crate::{Card, Player, State, NUM_CARDS};

/// Cards whose location is unknown to `searcher`: everything except the
/// searcher's current hand, every already-played card, and the revealed trump.
/// Path-determined (independent of any determinization), so it is also the
/// union of cards the opponent could possibly still hold.
pub fn unseen_cards(state: &State, searcher: Player) -> Vec<Card> {
    let mut seen = [false; NUM_CARDS];
    for c in state.hand(searcher) {
        seen[c.index()] = true;
    }
    for ev in &state.trick_history {
        seen[ev.card.index()] = true;
    }
    seen[state.trump.index()] = true;
    (0..NUM_CARDS)
        .filter(|&i| !seen[i])
        .map(Card::from_index)
        .collect()
}

/// Sample a full playable `State` consistent with `searcher`'s information set:
/// the searcher's hand / trump / history are kept; the opponent's hand is
/// resampled from the unseen pool respecting inferred voids (the 6 unused cards
/// are the remainder and are never referenced during play, so we don't store
/// them).
pub fn determinize<R: Rng + ?Sized>(state: &State, searcher: Player, rng: &mut R) -> State {
    let opp = searcher.other();
    let opp_count = state.hand(opp).len();
    let voids = opponent_voids(state, opp);

    // Split the unseen pool into cards forced into the unused pile (opponent-void
    // suits) and freely assignable cards.
    let mut forced_unused = 0usize;
    let mut free: Vec<Card> = Vec::with_capacity(opp_count + 6);
    for c in unseen_cards(state, searcher) {
        if voids[c.suit as usize] {
            forced_unused += 1;
        } else {
            free.push(c);
        }
    }
    debug_assert!(forced_unused <= 6, "void-suit cards cannot exceed the 6-card unused pile");

    // The unused pile takes `forced_unused` void cards plus `6 - forced_unused`
    // free cards; the remaining `opp_count` free cards become the opponent hand.
    let n_unused_free = 6 - forced_unused;
    fisher_yates(&mut free, rng);
    let mut opp_hand: Vec<Card> = free.split_off(n_unused_free);
    debug_assert_eq!(opp_hand.len(), opp_count, "opponent hand size mismatch");
    crate::sort_hand(&mut opp_hand);

    let mut s = state.clone();
    match opp {
        Player::Human => s.human_hand = opp_hand,
        Player::Bot => s.bot_hand = opp_hand,
    }
    s
}

fn fisher_yates<R: Rng + ?Sized>(xs: &mut [Card], rng: &mut R) {
    for i in (1..xs.len()).rev() {
        let j = rng.gen_range(0..=i);
        xs.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Phase, State};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Drive a match a few plies in, re-deriving `awaiting` each step, so we can
    /// determinize from non-trivial mid-round states.
    fn play_some(steps: usize, seed: u64) -> State {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut s = State::new_match(&mut rng);
        for _ in 0..steps {
            match s.phase {
                Phase::Playing => {
                    let legal = s.legal();
                    s.apply(legal[0]);
                }
                Phase::TrickComplete => s.advance_after_trick(),
                Phase::RoundOver => s.end_round(&mut rng),
                Phase::MatchOver => break,
            }
        }
        s
    }

    #[test]
    fn determinization_is_legal() {
        for seed in 0..200u64 {
            let s = play_some((seed % 20) as usize, seed);
            if s.phase != Phase::Playing {
                continue;
            }
            let searcher = s.awaiting.unwrap();
            let opp = searcher.other();
            let mut rng = StdRng::seed_from_u64(seed ^ 0xDEAD);
            let d = determinize(&s, searcher, &mut rng);

            // Searcher hand, trump, history, scores untouched.
            assert_eq!(d.hand(searcher), s.hand(searcher));
            assert_eq!(d.trump, s.trump);
            assert_eq!(d.trick_history.len(), s.trick_history.len());

            // Opponent hand: right size, no void-suit cards, no seen cards, distinct.
            let opp_hand = d.hand(opp);
            assert_eq!(opp_hand.len(), s.hand(opp).len());
            let voids = opponent_voids(&s, opp);
            let mut seen = [false; NUM_CARDS];
            for c in s.hand(searcher) {
                seen[c.index()] = true;
            }
            for ev in &s.trick_history {
                seen[ev.card.index()] = true;
            }
            seen[s.trump.index()] = true;
            let mut in_hand = [false; NUM_CARDS];
            for c in opp_hand {
                assert!(!voids[c.suit as usize], "dealt opponent a void-suit card");
                assert!(!seen[c.index()], "dealt opponent a seen card");
                assert!(!in_hand[c.index()], "duplicate card in opponent hand");
                in_hand[c.index()] = true;
            }
        }
    }
}
