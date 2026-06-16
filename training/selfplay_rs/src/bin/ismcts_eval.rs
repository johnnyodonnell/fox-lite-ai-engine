//! ismcts_eval — DISPOSABLE, run-it-occasionally test: does wrapping a snapshot
//! in ISMCTS search beat the same snapshot playing raw greedy policy?
//!
//! Side A = ISMCTS(model), Side B = greedy argmax(model) — SAME weights both
//! sides, so any edge is purely from search. Mirrored seating, win-rate + Elo CI
//! (same math as head_to_head.rs). Not wired into training/eval; delete with
//! `rm`. Defaults to `--device cuda`; pass `--device cpu` to keep off the GPU.
//! (When run on cuda, free the GPU by stopping the training daemon first.)
//!
//!   ismcts_eval --model <snap.safetensors> \
//!       [--games 200] [--sims 400] [--cpuct 1.5] [--seed 0] [--device cuda]
//!
//! Search unit is ONE round (round boundaries deal new cards = a chance node).
//! Leaf eval = the net's tanh value head (per-round z, mover's POV); round end
//! is an exact terminal. SO-ISMCTS: re-determinize the opponent's hidden hand
//! each simulation; a single tree keyed by the public card-play sequence;
//! per-determinization availability handled by only ranking currently-legal
//! actions. Determinizations respect revealed voids: a suit the opponent has
//! failed to follow is never dealt back into their sampled hand. Known limit:
//! optimizes per-round point margin (the value head's training target), not
//! match-win probability.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{
    encode, encode_v1, encode_v2, legal_mask, real_card_from_canon_index, INPUT_SIZE,
    INPUT_SIZE_V1, INPUT_SIZE_V2,
};
use foxlite_core::{
    sort_hand, score_for_tricks, Card, Phase, Player, State, NUM_CARDS, NUM_RANKS, NUM_SUITS,
};
use selfplay_rs::net::AnyNet;

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Thin wrapper: arch-aware encode + a single-state forward returning
/// (policy logits over 33 canonical slots, scalar tanh value from `mover`'s POV).
struct NetWrap {
    net: AnyNet,
}

impl NetWrap {
    fn arch(&self) -> &'static str {
        match self.net {
            AnyNet::V1(_) => "v1",
            AnyNet::V2(_) => "v2",
            AnyNet::V3(_) => "v3",
        }
    }

    fn eval(&self, state: &State, mover: Player) -> ([f32; NUM_CARDS], f32) {
        let (v, size) = match self.net {
            AnyNet::V1(_) => (encode_v1(state, mover), INPUT_SIZE_V1),
            AnyNet::V2(_) => (encode_v2(state, mover), INPUT_SIZE_V2),
            AnyNet::V3(_) => (encode(state, mover), INPUT_SIZE),
        };
        let x = Tensor::from_slice(&v)
            .reshape([1, size as i64])
            .to_device(self.net.device());
        let (logits, value) = self.net.forward(&x);
        let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
        let mut buf = [0f32; NUM_CARDS];
        logits.copy_data(&mut buf, NUM_CARDS);
        let val = value.to_kind(Kind::Float).double_value(&[0]) as f32;
        (buf, val)
    }
}

/// Greedy argmax over legal canonical slots (the raw-policy baseline).
fn greedy_act(net: &NetWrap, state: &State, mover: Player) -> usize {
    let mask = legal_mask(state, mover);
    let (logits, _v) = net.eval(state, mover);
    let mut best = usize::MAX;
    let mut best_v = f32::NEG_INFINITY;
    for j in 0..NUM_CARDS {
        if mask[j] != 0.0 && logits[j] > best_v {
            best_v = logits[j];
            best = j;
        }
    }
    best
}

// --- ISMCTS ---------------------------------------------------------------

struct Edge {
    n: u32,
    w: f64,     // sum of leaf values, ALWAYS from Human's POV (per-round z is antisymmetric)
    prior: f32, // net policy prior, set when the action's node was first expanded
    child: i32, // arena index, -1 if not yet created
}

struct Node {
    to_move: Player, // public (depends only on the action sequence), determinization-independent
    expanded: bool,
    edges: HashMap<u8, Edge>, // keyed by canonical card slot 0..33
}

/// Suits `player` has publicly revealed they hold none of: as a *follower* they
/// failed to follow the led suit (played off-suit), which is only legal when
/// void. Leading reveals nothing. `trick_history` is chronological with two
/// events per completed trick — the lead then the follow.
fn voided_suits(real: &State, player: Player) -> [bool; NUM_SUITS] {
    let mut void = [false; NUM_SUITS];
    let h = &real.trick_history;
    let mut i = 0;
    while i < h.len() {
        let lead = h[i];
        // Pair the lead with its follow (same trick). A lone trailing event is
        // the in-progress trick's lead — no void info there.
        if i + 1 < h.len() && h[i + 1].trick == lead.trick {
            let follow = h[i + 1];
            if follow.player == player && follow.card.suit != lead.card.suit {
                void[lead.card.suit as usize] = true;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    void
}

/// Sample a world consistent with `mover`'s information set: keep `mover`'s real
/// hand, trump and the played history; deal the opponent a random hand of the
/// (public) correct size from the unknown pool, EXCLUDING any suit the opponent
/// has shown a void in. The 6 unused cards fall out as whatever is left over.
fn determinize(real: &State, mover: Player, rng: &mut StdRng) -> State {
    let opp = mover.other();
    let mut known = [false; NUM_CARDS];
    for c in real.hand(mover) {
        known[c.index()] = true;
    }
    known[real.trump.index()] = true;
    for ev in &real.trick_history {
        known[ev.card.index()] = true;
    }
    let void = voided_suits(real, opp);
    // Cards the opponent could still hold: unknown to `mover` and not in a suit
    // the opponent has revealed a void in. Their real hand is a subset of this,
    // so there are always >= opp_size eligible cards.
    let mut pool: Vec<Card> = (0..NUM_CARDS)
        .filter(|i| !known[*i] && !void[*i / NUM_RANKS])
        .map(Card::from_index)
        .collect();
    pool.shuffle(rng);
    let opp_size = real.hand(opp).len();
    let mut opp_hand: Vec<Card> = pool[..opp_size].to_vec();
    sort_hand(&mut opp_hand);

    let mut s = real.clone();
    match opp {
        Player::Human => s.human_hand = opp_hand,
        Player::Bot => s.bot_hand = opp_hand,
    }
    s
}

/// Advance a just-applied state past any completed trick to the next decision
/// point (Playing) or round end (RoundOver).
fn settle(state: &mut State) {
    while state.phase == Phase::TrickComplete {
        state.advance_after_trick();
    }
}

/// One simulation: descend the shared tree under a fresh determinization,
/// expand one leaf, back the value up. `state` is owned and mutated down the path.
fn simulate(arena: &mut Vec<Node>, root: usize, mut state: State, net: &NetWrap, cpuct: f64) {
    let mut path: Vec<(usize, u8)> = Vec::new();
    let mut idx = root;
    let v_human: f32;

    loop {
        // Expand on first visit: priors (softmax over this determinization's
        // legal moves) + leaf value from the net.
        if !arena[idx].expanded {
            let p = arena[idx].to_move;
            let (logits, val) = net.eval(&state, p);
            let mask = legal_mask(&state, p);
            let legal: Vec<u8> = (0..NUM_CARDS).filter(|j| mask[*j] != 0.0).map(|j| j as u8).collect();
            let maxl = legal.iter().map(|a| logits[*a as usize]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for a in &legal {
                sum += (logits[*a as usize] - maxl).exp();
            }
            let mut edges = HashMap::new();
            for a in &legal {
                let pr = (logits[*a as usize] - maxl).exp() / sum.max(1e-9);
                edges.insert(*a, Edge { n: 0, w: 0.0, prior: pr, child: -1 });
            }
            arena[idx].edges = edges;
            arena[idx].expanded = true;
            v_human = if p == Player::Human { val } else { -val };
            break;
        }

        let p = arena[idx].to_move;
        let mask = legal_mask(&state, p);
        let legal: Vec<u8> = (0..NUM_CARDS).filter(|j| mask[*j] != 0.0).map(|j| j as u8).collect();
        // SO-ISMCTS availability: a card legal in this determinization but never
        // expanded (different opp hand) gets a lazy edge with a uniform prior.
        let unif = 1.0f32 / legal.len() as f32;
        for a in &legal {
            arena[idx].edges.entry(*a).or_insert(Edge { n: 0, w: 0.0, prior: unif, child: -1 });
        }
        let np: u32 = legal.iter().map(|a| arena[idx].edges[a].n).sum();
        let sqrt_np = ((np as f64) + 1.0).sqrt();

        // PUCT over currently-legal actions, from p's perspective.
        let mut best_a = legal[0];
        let mut best_score = f64::NEG_INFINITY;
        for a in &legal {
            let e = &arena[idx].edges[a];
            let q_human = if e.n > 0 { e.w / e.n as f64 } else { 0.0 };
            let q_p = if p == Player::Human { q_human } else { -q_human };
            let u = cpuct * e.prior as f64 * sqrt_np / (1.0 + e.n as f64);
            let score = q_p + u;
            if score > best_score {
                best_score = score;
                best_a = *a;
            }
        }

        let card = real_card_from_canon_index(best_a as usize, state.trump.suit);
        state.apply(card);
        settle(&mut state);
        path.push((idx, best_a));

        if state.phase == Phase::RoundOver {
            let hp = score_for_tricks(state.tricks_won[Player::Human.idx()]) as f32;
            let bp = score_for_tricks(state.tricks_won[Player::Bot.idx()]) as f32;
            v_human = (hp - bp) / 6.0;
            break;
        }

        let child = arena[idx].edges[&best_a].child;
        idx = if child < 0 {
            let nidx = arena.len();
            let to_move = state.awaiting.expect("Playing state has an awaiting player");
            arena.push(Node { to_move, expanded: false, edges: HashMap::new() });
            arena[idx].edges.get_mut(&best_a).unwrap().child = nidx as i32;
            nidx
        } else {
            child as usize
        };
    }

    for (node_idx, a) in path {
        let e = arena[node_idx].edges.get_mut(&a).unwrap();
        e.n += 1;
        e.w += v_human as f64;
    }
}

/// Run `sims` simulations from the real decision point and return the
/// most-visited root action (canonical slot).
fn ismcts_act(
    net: &NetWrap,
    real: &State,
    mover: Player,
    sims: usize,
    cpuct: f64,
    rng: &mut StdRng,
) -> usize {
    let mut arena: Vec<Node> = vec![Node { to_move: mover, expanded: false, edges: HashMap::new() }];
    for _ in 0..sims {
        let det = determinize(real, mover, rng);
        simulate(&mut arena, 0, det, net, cpuct);
    }
    arena[0]
        .edges
        .iter()
        .max_by_key(|(_, e)| e.n)
        .map(|(a, _)| *a as usize)
        .unwrap_or_else(|| greedy_act(net, real, mover))
}

/// One match: side A = ISMCTS, side B = greedy, both using `net`. Returns true
/// if A won. `a_is_human` controls seating for the mirror.
fn play_match(
    net: &NetWrap,
    a_is_human: bool,
    sims: usize,
    cpuct: f64,
    rng: &mut StdRng,
) -> bool {
    let mut s = State::new_match(rng);
    loop {
        match s.phase {
            Phase::Playing => {
                let mover = s.awaiting.unwrap();
                let a_to_move = (mover == Player::Human) == a_is_human;
                let action = if a_to_move {
                    ismcts_act(net, &s, mover, sims, cpuct, rng)
                } else {
                    greedy_act(net, &s, mover)
                };
                let card = real_card_from_canon_index(action, s.trump.suit);
                s.apply(card);
            }
            Phase::TrickComplete => s.advance_after_trick(),
            Phase::RoundOver => s.end_round(rng),
            Phase::MatchOver => {
                let w = s.match_winner().unwrap();
                return (w == Player::Human) == a_is_human;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = flag(&args, "--model", "");
    let games: usize = flag(&args, "--games", "200").parse().unwrap();
    let sims: usize = flag(&args, "--sims", "400").parse().unwrap();
    let cpuct: f64 = flag(&args, "--cpuct", "1.5").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    let device = flag(&args, "--device", "cuda");
    assert!(!model.is_empty(), "--model <snapshot.safetensors> required");

    let dev = match device.as_str() {
        "cuda" | "gpu" => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let net = NetWrap { net: AnyNet::load_auto(&model, dev, Kind::Float) };
    let stem = std::path::Path::new(&model)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model.clone());
    println!(
        "[ismcts] model={stem} ({}) games={games} sims={sims} cpuct={cpuct} device={dev:?}",
        net.arch()
    );
    println!("[ismcts] A=ISMCTS  B=greedy  (same weights both sides)");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a_wins = 0usize;
    for g in 0..games {
        let a_is_human = g % 2 == 0;
        if play_match(&net, a_is_human, sims, cpuct, &mut rng) {
            a_wins += 1;
        }
        if (g + 1) % 10 == 0 {
            println!("[ismcts] {}/{games}: ISMCTS {a_wins} - {} greedy", g + 1, g + 1 - a_wins);
        }
    }

    let p = a_wins as f64 / games as f64;
    println!(
        "[ismcts] result: ISMCTS {a_wins} - {} greedy ({:.1}%)",
        games - a_wins,
        100.0 * p
    );
    if p > 0.0 && p < 1.0 {
        let diff = 400.0 * (p / (1.0 - p)).log10();
        let se = (p * (1.0 - p) / games as f64).sqrt();
        let (lo, hi) = ((p - 1.96 * se).max(1e-9), (p + 1.96 * se).min(1.0 - 1e-9));
        let lo_d = 400.0 * (lo / (1.0 - lo)).log10();
        let hi_d = 400.0 * (hi / (1.0 - hi)).log10();
        println!("[ismcts] elo diff (ISMCTS - greedy): {diff:+.0} [95% CI {lo_d:+.0}, {hi_d:+.0}]");
    } else {
        println!("[ismcts] elo diff: sweep — outside Elo scale");
    }
}
