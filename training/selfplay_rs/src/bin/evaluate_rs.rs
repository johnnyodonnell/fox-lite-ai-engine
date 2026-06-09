//! evaluate_rs — Fox Lite evaluation loop, ported from chess-ai-engine's
//! `eval.rs`. A candidate snapshot plays match games against an *active pool*
//! chosen by rating: the top-`n_top` performers + a fixed `random` floor anchor
//! + `n_anchors` frozen snapshots whose ratings most evenly cover the Elo range
//! (0, rating of the n_top-th). A global Bradley-Terry Elo is then refit over
//! all accumulated match results (random pinned at 0), and pool.json is updated.
//!
//! Unlike chess there is no auto-serving — promotion to the browser model stays
//! manual (training/promote.py). Net agents play by ISMCTS search (`--sims`
//! simulations, root noise off, argmax-visit move); `random` is the floor anchor.
//!
//!   evaluate_rs --run-dir runs/run2 --candidate runs/run2/snapshots/snap_x.safetensors
//!               [--games 80] [--sims 400] [--n-top 2] [--n-anchors 3] [--seed 0]
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
use foxlite_core::mcts::{run_search, sample_move};
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
// Players / match play (Fox Lite domain: ISMCTS net agent + random, no draws)
// ---------------------------------------------------------------------------
enum Agent {
    /// Neural agent: ISMCTS search (`sims` simulations, noise off) from the
    /// mover's information set, picking the argmax-visit move.
    Net { net: Net, sims: usize },
    Random,
}

impl Agent {
    /// Choose a canonical card index for `mover` in `state`.
    fn act(&self, state: &State, mover: Player, rng: &mut StdRng) -> usize {
        match self {
            Agent::Random => {
                let mask = legal_mask(state, mover);
                let legal: Vec<usize> = (0..NUM_CARDS).filter(|&j| mask[j] != 0.0).collect();
                legal[rng.gen_range(0..legal.len())]
            }
            Agent::Net { net, sims } => {
                let arena = run_search(state, mover, *sims, false, rng, |s, m| {
                    let v = encode(s, m);
                    let x = Tensor::from_slice(&v)
                        .reshape([1, INPUT_SIZE as i64])
                        .to_device(net.device());
                    let (logits, value) = net.forward(&x);
                    let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
                    let mut lb = vec![0f32; NUM_CARDS];
                    logits.copy_data(&mut lb, NUM_CARDS);
                    let value = value.to_kind(Kind::Float).to_device(Device::Cpu).double_value(&[0]);
                    (lb, value)
                });
                sample_move(&arena, 0, 0.0, rng) // temperature 0 = argmax visits
            }
        }
    }
}

fn play_match(human: &Agent, bot: &Agent, rng: &mut StdRng) -> Player {
    let mut s = State::new_match(rng);
    loop {
        match s.phase {
            Phase::Playing => {
                let mover = s.awaiting.unwrap();
                let agent = if mover == Player::Human { human } else { bot };
                let action = agent.act(&s, mover, rng);
                let card = real_card_from_canon_index(action, s.trump.suit);
                s.apply(card);
            }
            Phase::TrickComplete => s.advance_after_trick(),
            Phase::RoundOver => s.end_round(rng),
            Phase::MatchOver => return s.match_winner().unwrap(),
        }
    }
}

/// Play `games` between `cand` and `opp`, mirroring seating to reduce
/// first-mover bias. Returns the candidate's win count (no draws in Fox Lite).
fn play_series(cand: &Agent, opp: &Agent, games: usize, rng: &mut StdRng) -> usize {
    let mut wins = 0usize;
    for g in 0..games {
        let cand_is_human = g % 2 == 0;
        let (human, bot) = if cand_is_human { (cand, opp) } else { (opp, cand) };
        let winner = play_match(human, bot, rng);
        if (winner == Player::Human) == cand_is_human {
            wins += 1;
        }
    }
    wins
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
    let sims: usize = flag(&args, "--sims", "400").parse().unwrap();
    let n_top: usize = flag(&args, "--n-top", "2").parse().unwrap();
    let n_anchors: usize = flag(&args, "--n-anchors", "3").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    assert!(candidate.as_os_str().len() > 0, "--candidate required");

    let dev = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

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

    let cand = Agent::Net {
        net: Net::load(candidate.to_str().expect("utf8 path"), dev, Kind::Float),
        sims,
    };

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
            Some(p) => Agent::Net {
                net: Net::load(run_dir.join(p).to_str().expect("utf8 path"), dev, Kind::Float),
                sims,
            },
        };
        let wins = play_series(&cand, &opp_agent, games, &mut rng);
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
