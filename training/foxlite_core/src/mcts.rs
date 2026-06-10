//! Information-Set MCTS (ISMCTS) with neural PUCT priors + value leaf eval.
//!
//! True ISMCTS: one shared statistics tree, re-determinized every simulation.
//! The tree is grown from the *searcher's* (root mover's) information set. A node
//! identified by its action path has a fixed, determinization-independent set of
//! *potential* children:
//!   - searcher-to-move node: the searcher's own legal moves (public);
//!   - opponent-to-move node: the legal moves over the *unseen* card set (the
//!     union of everything the opponent could still hold — also public, since the
//!     played path and the searcher's hand are fixed at a node).
//! Each simulation determinizes the hidden cards, descends selecting only among
//! children that are legal in *that* determinization (tracking per-child
//! availability), and bootstraps a value at the leaf from the net.
//!
//! Value bookkeeping is global ("searcher POV"): every node's `value_sum` is the
//! summed value from the searcher's perspective, so backprop just adds `v_ref`
//! with no sign juggling. The only place a sign appears is the PUCT exploit term,
//! where the node's mover maximizes its *own* value (`+q` if it is the searcher,
//! `-q` otherwise). This handles Fox-Lite's non-alternating turn order (a trick
//! winner leads again, so the mover can repeat) without chess's depth-parity flip.
//!
//! Search is bounded to the current round: a path that completes the round ends
//! the simulation. If the round ends the match → terminal value ±1; otherwise the
//! value is bootstrapped from a FRESH net eval of the state just before the
//! round-ending move, under the CURRENT determinization (`WalkResult::BoundaryEval`
//! — a frozen per-node value would belong to whichever hidden world first expanded
//! the node). So determinization only ever covers the current round's hidden cards.

use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

use crate::determinize::unseen_cards;
use crate::encode::{canon_card_index, real_card_from_canon_index};
use crate::{
    legal_moves, score_for_tricks, trick_winner, Phase, Player, Side, State, NUM_CARDS,
    TARGET_SCORE, TRICKS_PER_ROUND,
};

pub const C_PUCT: f64 = 1.5;
pub const DIRICHLET_ALPHA: f64 = 0.6; // higher than chess's 0.3 for Fox-Lite's ≤13 branching
pub const DIRICHLET_EPS: f64 = 0.25;
// Opponent-node priors are softmaxed from ONE determinization's net eval, and the
// policy head puts ~zero mass on cards outside the sampled hand — so children the
// first sampled hand didn't hold would keep ~zero prior forever and PUCT (whose
// explore term is prior-multiplied) would starve them even in determinizations
// where they are legal and strong. Blending toward uniform over the potential set
// guarantees every child an exploration floor; Q from visits then sorts them out.
pub const OPP_PRIOR_UNIFORM_EPS: f64 = 0.3;
pub const TEMP_OPENING: f64 = 1.0; // sampling temperature for the opening of a match
pub const TEMP_FLOOR: f64 = 0.25; // floor temperature once fully annealed
pub const TEMP_EARLY_TRICKS: u32 = 13; // match tricks held at the opening temperature (round 1)
pub const TEMP_ANNEAL_TRICKS: u32 = 26; // match tricks over which temperature ramps to the floor (rounds 2-3)

/// Self-play move-selection temperature, annealed over the whole match rather
/// than resetting each round. Holds at `TEMP_OPENING` through round 1, then
/// linearly anneals to `TEMP_FLOOR` across rounds 2-3. Round 4 is the earliest
/// a match can end (6 pts/round max toward 21), so every potentially deciding
/// round is played at the floor.
pub fn temperature(round_num: u32, trick_num: u32) -> f64 {
    let match_trick = (round_num - 1) * TRICKS_PER_ROUND + trick_num;
    if match_trick <= TEMP_EARLY_TRICKS {
        return TEMP_OPENING;
    }
    let frac = (((match_trick - TEMP_EARLY_TRICKS) as f64) / TEMP_ANNEAL_TRICKS as f64).min(1.0);
    TEMP_OPENING + frac * (TEMP_FLOOR - TEMP_OPENING)
}

pub struct Node {
    pub prior: f64,
    pub visit_count: u32,
    pub value_sum: f64, // searcher POV
    pub avail_count: u32,
    pub children: Vec<(u8, u32)>, // (canonical card index, child arena idx)
    pub expanded: bool,
    pub noised: bool,
    pub mover: Player, // seat to move at this node
}

impl Node {
    pub fn new(prior: f64, mover: Player) -> Node {
        Node {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            avail_count: 0,
            children: Vec::new(),
            expanded: false,
            noised: false,
            mover,
        }
    }
    fn q(&self) -> f64 {
        if self.visit_count > 0 {
            self.value_sum / self.visit_count as f64
        } else {
            0.0
        }
    }
}

/// A fresh root arena for searching as `searcher` (root index 0).
pub fn new_root(searcher: Player) -> Vec<Node> {
    vec![Node::new(0.0, searcher)]
}

/// Canonical indices legal for `mover` in `state` *right now* (this determinization).
fn legal_canon(state: &State, mover: Player) -> Vec<usize> {
    legal_moves(state.hand(mover), state.led_card)
        .iter()
        .map(|c| canon_card_index(*c, state.trump.suit))
        .collect()
}

/// Determinization-independent *potential* child set at a node (a superset of the
/// union over determinizations of the mover's legal moves):
///   - searcher node: the searcher's exact legal moves (public, deterministic);
///   - opponent node: *every* unseen card. We can't follow-suit-filter here — a
///     determinization that makes the opponent void in the led suit lets it play
///     off-suit, so any unseen card may become legal. Over-included cards that are
///     never legal in any determinization simply stay at zero availability and are
///     never selected.
fn potential_canon(state: &State, mover: Player, searcher: Player) -> Vec<usize> {
    let cards = if mover == searcher {
        legal_moves(state.hand(searcher), state.led_card)
    } else {
        unseen_cards(state, searcher)
    };
    cards
        .iter()
        .map(|c| canon_card_index(*c, state.trump.suit))
        .collect()
}

/// Seat to move after `mover` plays `canon` from `state` (placeholder for a
/// round-ending follow, which produces a terminal child whose mover is unused).
fn child_mover_after(state: &State, canon: usize, mover: Player) -> Player {
    match state.led_card {
        None => mover.other(), // leading — the other seat follows
        Some(led) => {
            if state.trick_num == TRICKS_PER_ROUND {
                mover // terminal child (round ends); placeholder
            } else {
                let card = real_card_from_canon_index(canon, state.trump.suit);
                match trick_winner(led, card, state.trump.suit) {
                    Side::Lead => state.leader,
                    Side::Follow => mover,
                }
            }
        }
    }
}

/// Match winner if a `RoundOver` state ends the match, else `None` (round
/// continues). Mirrors `State::end_round` scoring + `State::match_winner` ties,
/// read-only (no deal, no RNG) so search never crosses a round boundary.
fn round_over_outcome(state: &State) -> Option<Player> {
    let hl = score_for_tricks(state.tricks_won[Player::Human.idx()]);
    let bl = score_for_tricks(state.tricks_won[Player::Bot.idx()]);
    let nh = state.score[Player::Human.idx()] + hl;
    let nb = state.score[Player::Bot.idx()] + bl;
    if nh >= TARGET_SCORE || nb >= TARGET_SCORE {
        // Mirror `State::match_winner`: higher total wins; a tie is broken by the
        // final round's points, which can never themselves tie.
        if nh > nb {
            Some(Player::Human)
        } else if nb > nh {
            Some(Player::Bot)
        } else if hl > bl {
            Some(Player::Human)
        } else {
            Some(Player::Bot)
        }
    } else {
        None
    }
}

/// Bump availability for every currently-legal child, then pick the max-PUCT one.
/// Returns `(canonical idx, child arena idx)`.
fn select_child(arena: &mut [Node], node_idx: usize, avail: &[usize], searcher: Player) -> Option<(usize, usize)> {
    let mover = arena[node_idx].mover;
    let sign = if mover == searcher { 1.0 } else { -1.0 };
    let children = arena[node_idx].children.clone();
    for &(canon, ci) in &children {
        if avail.contains(&(canon as usize)) {
            arena[ci as usize].avail_count += 1;
        }
    }
    let mut best: Option<(usize, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for &(canon, ci) in &children {
        if !avail.contains(&(canon as usize)) {
            continue;
        }
        let c = &arena[ci as usize];
        let exploit = sign * c.q();
        let explore =
            C_PUCT * c.prior * (c.avail_count.max(1) as f64).sqrt() / (1.0 + c.visit_count as f64);
        let score = exploit + explore;
        if score > best_score {
            best_score = score;
            best = Some((canon as usize, ci as usize));
        }
    }
    best
}

/// Outcome of descending one simulation through the (re-determinized) tree.
pub enum WalkResult {
    /// Reached an unexpanded `Playing` node — `det` is left at it; eval + expand.
    Eval { path: Vec<usize>, mover: Player },
    /// Round ended without ending the match: bootstrap from a net eval of `det`
    /// (the state just before the round-ending move, under the CURRENT
    /// determinization, `mover` to play their forced last card). Backprop the
    /// value (mover POV) along `path`; no node is expanded.
    BoundaryEval { path: Vec<usize>, det: State, mover: Player },
    /// Path resolved (match over in-horizon) without needing a net eval.
    Terminal { path: Vec<usize>, v_ref: f64 },
}

/// Descend from `root` over the determinized `det`, applying selected moves, until
/// an unexpanded `Playing` leaf or a round/match-over terminal. Mutates `det`
/// (plays moves) and `arena` (availability counts).
pub fn walk_to_leaf(arena: &mut Vec<Node>, root: usize, det: &mut State, searcher: Player) -> WalkResult {
    let mut path = vec![root];
    let mut node_idx = root;
    loop {
        let mover = det.awaiting.expect("walk at a non-decision phase");
        let avail = legal_canon(det, mover);
        let (canon, child_idx) =
            select_child(arena, node_idx, &avail, searcher).expect("expanded node with no available child");
        let card = real_card_from_canon_index(canon, det.trump.suit);
        // Only the trick-13 follow can end the round; keep that pre-move state so
        // a non-match-ending boundary is re-evaluated under THIS determinization.
        let pre_move = if det.trick_num == TRICKS_PER_ROUND && det.led_card.is_some() {
            Some(det.clone())
        } else {
            None
        };
        det.apply(card);
        while det.phase == Phase::TrickComplete {
            det.advance_after_trick();
        }
        path.push(child_idx);
        match det.phase {
            Phase::Playing => {
                if arena[child_idx].expanded {
                    node_idx = child_idx;
                } else {
                    return WalkResult::Eval { path, mover: det.awaiting.unwrap() };
                }
            }
            Phase::RoundOver => {
                return match round_over_outcome(det) {
                    Some(winner) => {
                        let v_ref = if winner == searcher { 1.0 } else { -1.0 };
                        WalkResult::Terminal { path, v_ref }
                    }
                    None => WalkResult::BoundaryEval {
                        path,
                        det: pre_move.expect("round ended without a trick-13 follow"),
                        mover,
                    },
                };
            }
            Phase::MatchOver => unreachable!("search never deals a new round"),
            Phase::TrickComplete => unreachable!("trick completion drained above"),
        }
    }
}

/// Expand `node_idx`: create a child per *potential* move with a softmax prior
/// over the net logits.
pub fn expand_node(
    arena: &mut Vec<Node>,
    node_idx: usize,
    det: &State,
    logits: &[f32],
    searcher: Player,
) {
    let mover = arena[node_idx].mover;
    let p_canon = potential_canon(det, mover, searcher);
    if !p_canon.is_empty() {
        let maxl = p_canon.iter().map(|&i| logits[i] as f64).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = p_canon.iter().map(|&i| ((logits[i] as f64) - maxl).exp()).collect();
        let total: f64 = exps.iter().sum();
        let n = p_canon.len() as f64;
        for (k, &canon) in p_canon.iter().enumerate() {
            let mut prior = if total > 0.0 { exps[k] / total } else { 1.0 / n };
            if mover != searcher {
                // see OPP_PRIOR_UNIFORM_EPS: don't trust one sampled hand's softmax
                prior = (1.0 - OPP_PRIOR_UNIFORM_EPS) * prior + OPP_PRIOR_UNIFORM_EPS / n;
            }
            let cm = child_mover_after(det, canon, mover);
            let cidx = arena.len() as u32;
            arena.push(Node::new(prior, cm));
            arena[node_idx].children.push((canon as u8, cidx));
        }
    }
    arena[node_idx].expanded = true;
}

/// Add `v_ref` (searcher POV) to every node on the path.
pub fn backprop(arena: &mut [Node], path: &[usize], v_ref: f64) {
    for &idx in path {
        arena[idx].visit_count += 1;
        arena[idx].value_sum += v_ref;
    }
}

/// Idempotent Dirichlet noise on the root's child priors (self-play only).
pub fn add_dirichlet_noise<R: Rng + ?Sized>(arena: &mut [Node], root: usize, rng: &mut R) {
    if arena[root].noised || arena[root].children.is_empty() {
        return;
    }
    let child_idxs: Vec<u32> = arena[root].children.iter().map(|&(_, c)| c).collect();
    let k = child_idxs.len();
    let noise: Vec<f64> = if k == 1 {
        vec![1.0]
    } else {
        Dirichlet::new(&vec![DIRICHLET_ALPHA; k]).unwrap().sample(rng)
    };
    for (&ci, &nz) in child_idxs.iter().zip(noise.iter()) {
        let c = &mut arena[ci as usize];
        c.prior = (1.0 - DIRICHLET_EPS) * c.prior + DIRICHLET_EPS * nz;
    }
    arena[root].noised = true;
}

/// Root visit-count policy target over canonical card slots (length 33).
pub fn visits_to_pi(arena: &[Node], root: usize, temperature: f64) -> [f32; NUM_CARDS] {
    let mut pi = [0.0f32; NUM_CARDS];
    let children = &arena[root].children;
    if children.is_empty() {
        return pi;
    }
    let counts: Vec<f64> = children.iter().map(|&(_, c)| arena[c as usize].visit_count as f64).collect();
    let canon: Vec<usize> = children.iter().map(|&(ci, _)| ci as usize).collect();
    if temperature == 0.0 {
        pi[canon[argmax(&counts)]] = 1.0;
        return pi;
    }
    let powered: Vec<f64> = counts.iter().map(|&c| c.powf(1.0 / temperature)).collect();
    let total: f64 = powered.iter().sum();
    if total <= 0.0 {
        return pi;
    }
    for (&ci, &p) in canon.iter().zip(powered.iter()) {
        pi[ci] = (p / total) as f32;
    }
    pi
}

/// Sample a root move (canonical card index) from visit counts at `temperature`.
pub fn sample_move<R: Rng + ?Sized>(arena: &[Node], root: usize, temperature: f64, rng: &mut R) -> usize {
    let children = &arena[root].children;
    let counts: Vec<f64> = children.iter().map(|&(_, c)| arena[c as usize].visit_count as f64).collect();
    let canon: Vec<usize> = children.iter().map(|&(ci, _)| ci as usize).collect();
    if temperature == 0.0 {
        return canon[argmax(&counts)];
    }
    let powered: Vec<f64> = counts.iter().map(|&c| c.powf(1.0 / temperature)).collect();
    let total: f64 = powered.iter().sum();
    let r = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (i, &p) in powered.iter().enumerate() {
        acc += p;
        if r < acc {
            return canon[i];
        }
    }
    canon[canon.len() - 1]
}

fn argmax(xs: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// Synchronous ISMCTS search (used by the evaluator + tests; the self-play
/// pipeline drives the primitives above directly for leaf batching). `eval`
/// returns `(logits[33], value)` from the given mover's perspective. Returns the
/// arena (root at 0); read the move with `sample_move` / `visits_to_pi`.
pub fn run_search<R, F>(
    root_state: &State,
    searcher: Player,
    sims: usize,
    add_noise: bool,
    rng: &mut R,
    mut eval: F,
) -> Vec<Node>
where
    R: Rng + ?Sized,
    F: FnMut(&State, Player) -> (Vec<f32>, f64),
{
    let mut arena = new_root(searcher);
    let (logits, _value) = eval(root_state, searcher);
    expand_node(&mut arena, 0, root_state, &logits, searcher);
    if add_noise {
        add_dirichlet_noise(&mut arena, 0, rng);
    }
    for _ in 0..sims {
        let mut det = crate::determinize::determinize(root_state, searcher, rng);
        match walk_to_leaf(&mut arena, 0, &mut det, searcher) {
            WalkResult::Eval { path, mover } => {
                let leaf = *path.last().unwrap();
                let (logits, value) = eval(&det, mover);
                expand_node(&mut arena, leaf, &det, &logits, searcher);
                let v_ref = if mover == searcher { value } else { -value };
                backprop(&mut arena, &path, v_ref);
            }
            WalkResult::BoundaryEval { path, det, mover } => {
                let (_logits, value) = eval(&det, mover);
                let v_ref = if mover == searcher { value } else { -value };
                backprop(&mut arena, &path, v_ref);
            }
            WalkResult::Terminal { path, v_ref } => backprop(&mut arena, &path, v_ref),
        }
    }
    arena
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::legal_mask;
    use crate::{Phase, State};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Uniform dummy net: zero logits, neutral value.
    fn dummy_eval(_s: &State, _m: Player) -> (Vec<f32>, f64) {
        (vec![0.0f32; NUM_CARDS], 0.0)
    }

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
    fn search_returns_legal_move_and_valid_pi() {
        for seed in 0..120u64 {
            let s = play_some((seed % 23) as usize, seed);
            if s.phase != Phase::Playing {
                continue;
            }
            let searcher = s.awaiting.unwrap();
            let mut rng = StdRng::seed_from_u64(seed ^ 0xABCD);
            let arena = run_search(&s, searcher, 64, true, &mut rng, dummy_eval);

            let mask = legal_mask(&s, searcher);
            let pi = visits_to_pi(&arena, 0, 1.0);

            // pi mass only on legal slots, sums to ~1.
            let mut total = 0.0f32;
            for j in 0..NUM_CARDS {
                if pi[j] > 0.0 {
                    assert!(mask[j] != 0.0, "pi mass on an illegal move (seed {seed})");
                }
                total += pi[j];
            }
            assert!((total - 1.0).abs() < 1e-4, "pi must sum to 1 (got {total}, seed {seed})");

            // sampled + argmax moves are legal.
            let mv = sample_move(&arena, 0, 1.0, &mut rng);
            assert!(mask[mv] != 0.0, "sampled illegal move (seed {seed})");
            let best = sample_move(&arena, 0, 0.0, &mut rng);
            assert!(mask[best] != 0.0, "argmax illegal move (seed {seed})");
        }
    }

    #[test]
    fn root_visits_match_sim_count() {
        let s = play_some(3, 7);
        let searcher = s.awaiting.unwrap();
        let mut rng = StdRng::seed_from_u64(99);
        let sims = 50;
        let arena = run_search(&s, searcher, sims, false, &mut rng, dummy_eval);
        // Every simulation backprops through the root exactly once.
        assert_eq!(arena[0].visit_count as usize, sims);
        // Child visit counts sum to the simulation count (each sim picks one root child).
        let child_visits: u32 = arena[0].children.iter().map(|&(_, c)| arena[c as usize].visit_count).sum();
        assert_eq!(child_visits as usize, sims);
    }
}
