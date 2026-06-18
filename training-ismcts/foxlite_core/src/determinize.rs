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

use crate::encode::{canon_card_index, opponent_voids};
use crate::{Card, Player, State, NUM_CARDS};

/// Floor for belief weights so a confident-zero card still gets a finite ES key
/// (`1/w` would otherwise blow up) and can never be force-excluded from sampling.
const BELIEF_EPS: f64 = 1e-6;

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
///
/// `belief` (optional) is a per-canonical-slot weight in [0,1] ≈ P(opponent holds
/// the card); when supplied, the opponent hand is drawn by weighted sampling
/// without replacement instead of uniformly. See [`determinize_into`].
pub fn determinize<R: Rng + ?Sized>(
    state: &State,
    searcher: Player,
    rng: &mut R,
    belief: Option<&[f32; NUM_CARDS]>,
) -> State {
    let mut out = state.clone();
    determinize_into(state, searcher, rng, &mut out, belief);
    out
}

/// [`determinize`] into a caller-owned scratch `State`, reusing its heap buffers
/// (`clone_from`) — the self-play pipeline determinizes once per simulation, and
/// the fresh `State` (3 Vec clones) plus unseen/free/hand Vecs per call were a
/// top malloc-churn source. Draws the same RNG sequence as the original.
pub fn determinize_into<R: Rng + ?Sized>(
    state: &State,
    searcher: Player,
    rng: &mut R,
    out: &mut State,
    belief: Option<&[f32; NUM_CARDS]>,
) {
    let opp = searcher.other();
    let opp_count = state.hand(opp).len();
    let voids = opponent_voids(state, opp);

    // Unseen pool, split into cards forced into the unused pile (opponent-void
    // suits) and freely assignable cards — fused with the seen-mask scan so no
    // intermediate Vec is built (cf. `unseen_cards`).
    let mut seen = [false; NUM_CARDS];
    for c in state.hand(searcher) {
        seen[c.index()] = true;
    }
    for ev in &state.trick_history {
        seen[ev.card.index()] = true;
    }
    seen[state.trump.index()] = true;
    let mut free = [Card::new(0, 1); NUM_CARDS];
    let mut n_free = 0usize;
    let mut forced_unused = 0usize;
    for i in 0..NUM_CARDS {
        if seen[i] {
            continue;
        }
        let c = Card::from_index(i);
        if voids[c.suit as usize] {
            forced_unused += 1;
        } else {
            free[n_free] = c;
            n_free += 1;
        }
    }
    debug_assert!(forced_unused <= 6, "void-suit cards cannot exceed the 6-card unused pile");

    // The unused pile takes `forced_unused` void cards plus `6 - forced_unused`
    // free cards; the remaining `opp_count` free cards become the opponent hand.
    let n_unused_free = 6 - forced_unused;
    debug_assert_eq!(n_free - n_unused_free, opp_count, "opponent hand size mismatch");

    out.clone_from(state);
    let trump = state.trump.suit;
    let opp_hand = match opp {
        Player::Human => &mut out.human_hand,
        Player::Bot => &mut out.bot_hand,
    };
    opp_hand.clear();
    match belief {
        // Uniform: shuffle and take the last `opp_count` free cards.
        None => {
            fisher_yates(&mut free[..n_free], rng);
            opp_hand.extend_from_slice(&free[n_unused_free..n_free]);
        }
        // Belief-weighted: Efraimidis–Spirakis weighted sampling without
        // replacement. Each free card gets key g = ln(u)/w (u~Uniform(0,1],
        // w=belief weight floored to eps); the `opp_count` largest-key cards are
        // the opponent hand, the rest are the unused pile. Higher weight ⇒ key
        // closer to 0 ⇒ more likely selected.
        Some(w) => {
            let mut keyed = [(0.0f64, Card::new(0, 1)); NUM_CARDS];
            for (i, &c) in free[..n_free].iter().enumerate() {
                let wi = (w[canon_card_index(c, trump)] as f64).clamp(BELIEF_EPS, 1.0);
                let u = rng.gen::<f64>().max(1e-12);
                keyed[i] = (u.ln() / wi, c);
            }
            keyed[..n_free].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(_, c) in &keyed[..opp_count] {
                opp_hand.push(c);
            }
        }
    }
    crate::sort_hand(opp_hand);
}

/// Sentinel-encoded opponent-hand belief *target* over canonical card slots, from
/// `searcher`'s POV (the supervised label for the net's belief head in self-play):
///   `1.0`  = opponent currently holds this card,
///   `0.0`  = card is unseen but not held (i.e. in the 6-card unused pile),
///   `-1.0` = card is seen (searcher's hand / already played / trump) → masked.
/// The mask (`>= 0`) is exactly the unseen set the determinizer samples over, so the
/// label and the sampler agree card-for-card.
pub fn opponent_belief_target(state: &State, searcher: Player) -> [f32; NUM_CARDS] {
    let opp = searcher.other();
    let trump = state.trump.suit;
    let mut seen = [false; NUM_CARDS];
    for c in state.hand(searcher) {
        seen[c.index()] = true;
    }
    for ev in &state.trick_history {
        seen[ev.card.index()] = true;
    }
    seen[state.trump.index()] = true;
    let mut opp_has = [false; NUM_CARDS];
    for c in state.hand(opp) {
        opp_has[c.index()] = true;
    }
    let mut tgt = [-1.0f32; NUM_CARDS];
    for i in 0..NUM_CARDS {
        if seen[i] {
            continue; // masked: location known to the searcher
        }
        let c = Card::from_index(i);
        tgt[canon_card_index(c, trump)] = if opp_has[i] { 1.0 } else { 0.0 };
    }
    tgt
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
            let d = determinize(&s, searcher, &mut rng, None);

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

    /// Belief-weighted determinization must still produce a legal hidden world
    /// (right size, no void-suit/seen/duplicate cards), and must over-sample
    /// high-weight cards relative to uniform.
    #[test]
    fn weighted_determinization_is_legal_and_biased() {
        // Find a mid-round state where the opponent has unseen cards to sample.
        let s = (0..200u64)
            .map(|seed| play_some((seed % 20) as usize + 3, seed))
            .find(|s| {
                s.phase == Phase::Playing
                    && s.awaiting.is_some()
                    && !s.hand(s.awaiting.unwrap().other()).is_empty()
            })
            .expect("a playable mid-round state");
        let searcher = s.awaiting.unwrap();
        let opp = searcher.other();
        let trump = s.trump.suit;

        // Build a belief that strongly favors the opponent's true cards, so a
        // biased sampler should deal them far more often than uniform would.
        let mut belief = [0.05f32; NUM_CARDS];
        let true_slots: Vec<usize> =
            s.hand(opp).iter().map(|c| canon_card_index(*c, trump)).collect();
        for &slot in &true_slots {
            belief[slot] = 0.95;
        }

        let mut rng = StdRng::seed_from_u64(0xF00D);
        let voids = opponent_voids(&s, opp);
        let mut seen = [false; NUM_CARDS];
        for c in s.hand(searcher) {
            seen[c.index()] = true;
        }
        for ev in &s.trick_history {
            seen[ev.card.index()] = true;
        }
        seen[s.trump.index()] = true;

        let trials = 400;
        let mut weighted_hits = 0u32;
        let mut uniform_hits = 0u32;
        for _ in 0..trials {
            let d = determinize(&s, searcher, &mut rng, Some(&belief));
            for c in d.hand(opp) {
                assert!(!voids[c.suit as usize], "void-suit card dealt");
                assert!(!seen[c.index()], "seen card dealt");
                if s.hand(opp).contains(c) {
                    weighted_hits += 1;
                }
            }
            assert_eq!(d.hand(opp).len(), s.hand(opp).len(), "hand size changed");

            let u = determinize(&s, searcher, &mut rng, None);
            for c in u.hand(opp) {
                if s.hand(opp).contains(c) {
                    uniform_hits += 1;
                }
            }
        }
        // The biased sampler should recover the true cards more often than uniform.
        assert!(
            weighted_hits > uniform_hits,
            "belief weighting did not bias toward true cards (weighted={weighted_hits}, uniform={uniform_hits})"
        );
    }

    /// The belief target masks exactly the seen cards and labels every unseen card
    /// by whether the opponent actually holds it.
    #[test]
    fn belief_target_matches_truth() {
        for seed in 0..200u64 {
            let s = play_some((seed % 20) as usize, seed);
            if s.phase != Phase::Playing {
                continue;
            }
            let searcher = s.awaiting.unwrap();
            let opp = searcher.other();
            let trump = s.trump.suit;
            let tgt = opponent_belief_target(&s, searcher);

            let mut seen = [false; NUM_CARDS];
            for c in s.hand(searcher) {
                seen[c.index()] = true;
            }
            for ev in &s.trick_history {
                seen[ev.card.index()] = true;
            }
            seen[s.trump.index()] = true;

            for i in 0..NUM_CARDS {
                let slot = canon_card_index(Card::from_index(i), trump);
                if seen[i] {
                    assert_eq!(tgt[slot], -1.0, "seen card not masked (seed {seed})");
                } else {
                    let held = s.hand(opp).iter().any(|c| c.index() == i);
                    assert_eq!(tgt[slot], if held { 1.0 } else { 0.0 }, "wrong label (seed {seed})");
                }
            }
        }
    }
}
