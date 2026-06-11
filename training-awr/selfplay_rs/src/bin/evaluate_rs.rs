//! evaluate_rs — Fox Lite evaluation loop, ported from chess-ai-engine's
//! `eval.rs`. A candidate snapshot plays match games against an *active pool*
//! chosen by rating: the top-`n_top` performers + a fixed `random` floor anchor
//! + `n_anchors` frozen snapshots whose ratings most evenly cover the Elo range
//! (0, rating of the n_top-th). A global Bradley-Terry Elo is then refit over
//! all accumulated match results (random pinned at 0), and pool.json is updated.
//!
//! Unlike chess there is no auto-serving — promotion to the browser model stays
//! manual (training/promote.py). Net agents play the **raw policy**: one
//! batched forward per step, argmax over the legal-masked logits — exactly the
//! deployed browser engine's behavior. No search; `random` is the floor anchor.
//!
//!   evaluate_rs --run-dir runs/run4 --candidate runs/run4/snapshots/snap_x.safetensors
//!               [--games 80] [--n-top 2] [--n-anchors 3] [--seed 0]
//!
//! stdout: human-readable `[eval]` logs (opponents, per-pair results, ratings).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{encode, legal_mask, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::{Phase, Player, State, NUM_CARDS};
use selfplay_rs::net::Net;

const RANDOM: &str = "random"; // reserved name for the fixed floor anchor (Elo pinned at 0)

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ---------------------------------------------------------------------------
// Net evaluation (batched fp32 forward over a stack of encodings)
// ---------------------------------------------------------------------------
/// Run `net` over `m` encodings (`enc` is `m * INPUT_SIZE` f32) and return the
/// policy logits (`m * NUM_CARDS`) on the host. The value head is unused.
fn eval_batch(net: &Net, enc: &[f32], m: usize) -> Vec<f32> {
    let x = Tensor::from_slice(enc)
        .reshape([m as i64, INPUT_SIZE as i64])
        .to_device(net.device());
    let (logits, _values) = tch::no_grad(|| net.forward(&x));
    let lc = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
    let mut lv = vec![0f32; m * NUM_CARDS];
    lc.copy_data(&mut lv, m * NUM_CARDS);
    lv
}

/// Argmax of `logits` over the slots where `mask` is set (the raw-policy move;
/// softmax is monotone, so masking + argmax on logits is exact).
fn argmax_legal(logits: &[f32], mask: &[f32; NUM_CARDS]) -> usize {
    let mut best = usize::MAX;
    let mut best_l = f32::NEG_INFINITY;
    for i in 0..NUM_CARDS {
        if mask[i] != 0.0 && logits[i] > best_l {
            best_l = logits[i];
            best = i;
        }
    }
    assert!(best != usize::MAX, "no legal move");
    best
}

// ---------------------------------------------------------------------------
// Players / match play (Fox Lite domain: raw-policy net agent + random, no draws)
// ---------------------------------------------------------------------------
enum Agent {
    /// Neural agent: raw policy — argmax over the legal-masked logits of a
    /// single forward, matching the deployed browser engine (no search).
    Net(Net),
    Random,
}

impl Agent {
    /// Choose a canonical card index for every `(state, searcher)` pair where
    /// this agent is to move. Net agents stack every position into ONE forward
    /// so the GPU work spans every concurrent game, not one at a time.
    fn select_moves(&self, states: &[&State], searchers: &[Player], rng: &mut StdRng) -> Vec<usize> {
        match self {
            Agent::Random => states
                .iter()
                .zip(searchers)
                .map(|(s, &mover)| {
                    let mask = legal_mask(s, mover);
                    let legal: Vec<usize> = (0..NUM_CARDS).filter(|&j| mask[j] != 0.0).collect();
                    legal[rng.gen_range(0..legal.len())]
                })
                .collect(),
            Agent::Net(net) => {
                let m = states.len();
                let mut enc = vec![0f32; m * INPUT_SIZE];
                for (j, (&s, &mover)) in states.iter().zip(searchers).enumerate() {
                    let e = encode(s, mover);
                    enc[j * INPUT_SIZE..(j + 1) * INPUT_SIZE].copy_from_slice(&e);
                }
                let logits = eval_batch(net, &enc, m);
                states
                    .iter()
                    .zip(searchers)
                    .enumerate()
                    .map(|(j, (&s, &mover))| {
                        let mask = legal_mask(s, mover);
                        argmax_legal(&logits[j * NUM_CARDS..(j + 1) * NUM_CARDS], &mask)
                    })
                    .collect()
            }
        }
    }
}

/// Play `games` between `cand` and `opp`, mirroring seating to reduce first-mover
/// bias. All games run concurrently; each step batches the net forwards across
/// every game where the same agent is to move. Returns the candidate's win count
/// (no draws in Fox Lite).
fn play_match(cand: &Agent, opp: &Agent, games: usize, rng: &mut StdRng) -> usize {
    struct MatchGame {
        state: State,
        cand_is_human: bool, // candidate plays the Human seat in this game
        done: bool,
        winner: Option<Player>,
    }

    let mut gs: Vec<MatchGame> = (0..games)
        .map(|g| MatchGame {
            state: State::new_match(rng),
            cand_is_human: g % 2 == 0,
            done: false,
            winner: None,
        })
        .collect();

    while gs.iter().any(|g| !g.done) {
        // Partition active games by which agent is to move this step. A game is
        // in exactly one group, so playing one group never disturbs the other.
        let mut cand_idx: Vec<usize> = Vec::new();
        let mut opp_idx: Vec<usize> = Vec::new();
        for (gi, g) in gs.iter().enumerate() {
            if g.done {
                continue;
            }
            let mover = g.state.awaiting.expect("active game awaits a mover");
            if (mover == Player::Human) == g.cand_is_human {
                cand_idx.push(gi);
            } else {
                opp_idx.push(gi);
            }
        }

        for (agent, group) in [(cand, &cand_idx), (opp, &opp_idx)] {
            if group.is_empty() {
                continue;
            }
            // Scope the immutable borrow of `gs` so we can mutate it below.
            let moves = {
                let states: Vec<&State> = group.iter().map(|&gi| &gs[gi].state).collect();
                let searchers: Vec<Player> =
                    group.iter().map(|&gi| gs[gi].state.awaiting.unwrap()).collect();
                agent.select_moves(&states, &searchers, rng)
            };
            for (k, &gi) in group.iter().enumerate() {
                let card = real_card_from_canon_index(moves[k], gs[gi].state.trump.suit);
                gs[gi].state.apply(card);
                loop {
                    match gs[gi].state.phase {
                        Phase::Playing | Phase::MatchOver => break,
                        Phase::TrickComplete => gs[gi].state.advance_after_trick(),
                        Phase::RoundOver => gs[gi].state.end_round(rng),
                    }
                }
                if gs[gi].state.phase == Phase::MatchOver {
                    gs[gi].winner = gs[gi].state.match_winner();
                    gs[gi].done = true;
                }
            }
        }
    }

    gs.iter()
        .filter(|g| (g.winner == Some(Player::Human)) == g.cand_is_human)
        .count()
}

// ---------------------------------------------------------------------------
// Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized)
// ---------------------------------------------------------------------------
/// `games`: (a, b, score_a, n) — in n games between a and b, a scored score_a
/// points (win=1). Names in `fixed` are held at their fixed rating to pin the
/// scale; `reg` pulls ratings toward 0 so a 100% sweep stays finite.
fn fit_elo(
    names: &[String],
    games: &[(String, String, f64, i64)],
    fixed: &HashMap<String, f64>,
    reg: f64,
    iters: usize,
) -> HashMap<String, f64> {
    let mut r: HashMap<String, f64> =
        names.iter().map(|n| (n.clone(), *fixed.get(n).unwrap_or(&0.0))).collect();
    let q = 10f64.ln() / 400.0;

    // adjacency: name -> Vec<(opponent, score_for_name, n)>
    let mut adj: HashMap<String, Vec<(String, f64, f64)>> =
        names.iter().map(|n| (n.clone(), Vec::new())).collect();
    for (a, b, score_a, n) in games {
        if *n <= 0 {
            continue;
        }
        let nf = *n as f64;
        adj.get_mut(a).unwrap().push((b.clone(), *score_a, nf));
        adj.get_mut(b).unwrap().push((a.clone(), nf - *score_a, nf));
    }

    for _ in 0..iters {
        for p in names {
            if fixed.contains_key(p) {
                continue;
            }
            let mut g = 0.0;
            let mut h = 0.0;
            let rp = r[p];
            for (opp, score_p, n) in &adj[p] {
                let e = 1.0 / (1.0 + 10f64.powf((r[opp] - rp) / 400.0));
                g += q * (score_p - n * e);
                h += q * q * n * e * (1.0 - e);
            }
            g -= reg * rp;
            h += reg;
            if h > 1e-12 {
                *r.get_mut(p).unwrap() += g / h;
            }
        }
    }
    for (n, v) in fixed {
        r.insert(n.clone(), *v);
    }
    r
}

// ---------------------------------------------------------------------------
// Pool (on-disk JSON). No serving fields — promotion stays manual.
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone, Default)]
struct ModelEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    st: Option<String>, // safetensors path relative to run-dir (None for random)
    #[serde(default)]
    rating: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct MatchResult {
    a: String,
    b: String,
    score_a: f64,
    n: i64,
}

#[derive(Serialize, Deserialize, Default)]
struct Pool {
    #[serde(default)]
    models: IndexMap<String, ModelEntry>,
    #[serde(default)]
    results: Vec<MatchResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    top: Vec<String>,
}

fn load_pool(path: &Path) -> Pool {
    if path.exists() {
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
    } else {
        Pool::default()
    }
}

fn save_pool(path: &Path, pool: &Pool) {
    let tmp = path.with_extension("tmp");
    let text = serde_json::to_string_pretty(pool).expect("serialize pool");
    fs::write(&tmp, text).unwrap_or_else(|e| panic!("write {tmp:?}: {e}"));
    fs::rename(&tmp, path).unwrap_or_else(|e| panic!("rename {tmp:?} -> {path:?}: {e}"));
}

/// Active opponents = top `n_top` rated + random + `n_anchors` frozen snapshots
/// whose ratings most evenly cover (0, rating of the n_top-th). Returns
/// (active opponent set, top list). `models` must be in pool insertion order.
fn select_anchors(
    ratings: &HashMap<String, f64>,
    models: &[String],
    n_top: usize,
    n_anchors: usize,
) -> (Vec<String>, Vec<String>) {
    let rget = |m: &str| ratings.get(m).copied().unwrap_or(0.0);

    let mut ranked: Vec<String> = models.iter().filter(|m| *m != RANDOM).cloned().collect();
    // Stable descending sort by rating (ties keep insertion order, like Python).
    ranked.sort_by(|a, b| rget(b).partial_cmp(&rget(a)).unwrap());

    let top: Vec<String> = ranked.iter().take(n_top).cloned().collect();
    let mut active = top.clone();
    active.push(RANDOM.to_string());

    let below: Vec<String> = ranked.iter().skip(n_top).cloned().collect();
    if !below.is_empty() {
        let ceiling = if let Some(last_top) = top.last() {
            rget(last_top)
        } else {
            below.iter().map(|m| rget(m)).fold(f64::NEG_INFINITY, f64::max)
        };
        let k = n_anchors.min(below.len());
        let mut pool: Vec<String> = below;
        for i in 0..k {
            let t = ceiling * (i + 1) as f64 / (k + 1) as f64;
            // Closest unused model (first on ties, matching Python min()).
            let bi = pool
                .iter()
                .enumerate()
                .min_by(|(_, m1), (_, m2)| {
                    (rget(m1) - t).abs().partial_cmp(&(rget(m2) - t).abs()).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();
            active.push(pool.remove(bi));
        }
    }
    (active, top)
}

fn snapshot_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = PathBuf::from(flag(&args, "--run-dir", "runs/run1"));
    let candidate = PathBuf::from(flag(&args, "--candidate", ""));
    let games: usize = flag(&args, "--games", "200").parse().unwrap();
    let n_top: usize = flag(&args, "--n-top", "2").parse().unwrap();
    let n_anchors: usize = flag(&args, "--n-anchors", "3").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    assert!(candidate.as_os_str().len() > 0, "--candidate required");

    let dev = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    // One-off A/B mode: --vs <safetensors|random> plays candidate against a
    // single opponent (both raw-policy argmax) and prints the result without
    // touching pool.json.
    let vs = flag(&args, "--vs", "");
    if !vs.is_empty() {
        let mut rng = StdRng::seed_from_u64(seed);
        let cand_name = snapshot_stem(&candidate);
        let cand = Agent::Net(Net::load(candidate.to_str().expect("utf8 path"), dev, Kind::Float));
        let opp_agent = if vs == RANDOM {
            Agent::Random
        } else {
            Agent::Net(Net::load(&vs, dev, Kind::Float))
        };
        let wins = play_match(&cand, &opp_agent, games, &mut rng);
        println!(
            "[ab] {cand_name} vs {}: {wins}/{games} ({:.1}%)",
            snapshot_stem(Path::new(&vs)),
            100.0 * wins as f64 / games.max(1) as f64
        );
        return;
    }

    let pool_path = run_dir.join("pool.json");
    let mut pool = load_pool(&pool_path);
    let mut rng = StdRng::seed_from_u64(seed);

    let cand_name = snapshot_stem(&candidate);
    println!("[eval] candidate={cand_name} device={dev:?}");

    // Register candidate (path stored relative to run-dir for portability) and
    // the fixed random floor anchor.
    let rel = |p: &Path| -> String {
        p.strip_prefix(&run_dir)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string_lossy().into_owned())
    };
    pool.models.insert(
        cand_name.clone(),
        ModelEntry { st: Some(rel(&candidate)), rating: None },
    );
    pool.models
        .entry(RANDOM.to_string())
        .or_insert(ModelEntry { st: None, rating: Some(0.0) });

    // Pick the active opponent set from prior ratings (top-N + random + anchors).
    let prior: HashMap<String, f64> =
        pool.models.iter().map(|(m, e)| (m.clone(), e.rating.unwrap_or(0.0))).collect();
    let model_names: Vec<String> = pool.models.keys().cloned().collect();
    let (active, _) = select_anchors(&prior, &model_names, n_top, n_anchors);
    let mut opponents: Vec<String> = active.into_iter().filter(|m| *m != cand_name).collect();
    if opponents.is_empty() {
        opponents = vec![RANDOM.to_string()];
    }
    println!("[eval] opponents={opponents:?}");

    let cand = Agent::Net(Net::load(candidate.to_str().expect("utf8 path"), dev, Kind::Float));

    // Resolve each opponent's safetensors path up front so the match loop can
    // mutate pool.results without holding a borrow on the pool.
    let opp_specs: Vec<(String, Option<String>)> = opponents
        .iter()
        .map(|o| {
            let st = if o == RANDOM {
                None
            } else {
                Some(pool.models[o].st.clone().unwrap_or_else(|| panic!("model {o} has no st path")))
            };
            (o.clone(), st)
        })
        .collect();

    let mut cand_vs_random: Option<f64> = None;
    for (opp, st) in &opp_specs {
        let opp_agent = match st {
            None => Agent::Random,
            Some(p) => Agent::Net(Net::load(run_dir.join(p).to_str().expect("utf8 path"), dev, Kind::Float)),
        };
        let wins = play_match(&cand, &opp_agent, games, &mut rng);
        pool.results.push(MatchResult {
            a: cand_name.clone(),
            b: opp.clone(),
            score_a: wins as f64,
            n: games as i64,
        });
        if opp == RANDOM {
            cand_vs_random = Some(wins as f64 / games.max(1) as f64);
        }
        println!(
            "[eval] {cand_name} vs {opp}: {wins}/{games} ({:.1}%)",
            100.0 * wins as f64 / games.max(1) as f64
        );
    }

    // Refit Elo globally over all accumulated results (random pinned at 0).
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let game_rows: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &game_rows, &fixed, 1e-4, 400);
    for m in &names {
        pool.models.get_mut(m).unwrap().rating = Some(round1(ratings[m]));
    }

    // Record the top list (used to seed next round's anchor selection).
    let (_, top) = select_anchors(&ratings, &names, n_top, n_anchors);
    pool.top = top;

    let mut ranked = names.clone();
    ranked.sort_by(|a, b| ratings[b].partial_cmp(&ratings[a]).unwrap());
    let rating_str = ranked
        .iter()
        .map(|m| format!("{m}={}", pool.models[m].rating.unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[eval] ratings: {rating_str}");

    let wr = cand_vs_random.map(|c| 100.0 * c).unwrap_or(f64::NAN);
    println!(
        "[eval] {cand_name} elo={:.0} vs_random={:.1}%",
        ratings[&cand_name], wr
    );

    save_pool(&pool_path, &pool);
    println!("[eval] done");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Play `steps` legal-first moves from a fresh match to reach a varied state.
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

    /// argmax_legal must pick the max-logit LEGAL slot over real game states,
    /// never an illegal one — even when an illegal slot holds the global max.
    #[test]
    fn argmax_legal_picks_max_legal_slot() {
        let mut rng = StdRng::seed_from_u64(99);
        for seed in 0..120u64 {
            let s = play_some((seed % 23) as usize, seed);
            if s.phase != Phase::Playing {
                continue;
            }
            let mover = s.awaiting.unwrap();
            let mask = legal_mask(&s, mover);
            let logits: Vec<f32> = (0..NUM_CARDS).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();
            let pick = argmax_legal(&logits, &mask);
            assert!(mask[pick] != 0.0, "picked an illegal slot (seed {seed})");
            for j in 0..NUM_CARDS {
                if mask[j] != 0.0 {
                    assert!(logits[pick] >= logits[j], "not the max legal logit (seed {seed})");
                }
            }
        }
    }
}
