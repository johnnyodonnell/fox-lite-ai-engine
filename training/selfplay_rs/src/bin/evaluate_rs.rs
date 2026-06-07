//! evaluate_rs — play a candidate snapshot vs a pool (random + recent prior
//! snapshots), greedy (argmax) policy, and report win counts as JSON on stdout.
//! Elo bookkeeping is done by the caller (training/elo.py).
//!
//!   evaluate_rs --run-dir runs/run1 --candidate runs/run1/snapshots/snap_x.safetensors
//!               [--games 200] [--pool-size 6] [--seed 0]
//!
//! stdout: {"candidate":"snap_x","results":[{"opponent":"random","games":G,"wins":W},...]}
//! stderr: progress logs.

use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{encode, legal_mask, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::{Phase, Player, State, NUM_CARDS};
use selfplay_rs::net::Net;

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

enum Agent {
    Net(Net),
    Random,
}

impl Agent {
    /// Choose a canonical card index for `mover` in `state`.
    fn act(&self, state: &State, mover: Player, rng: &mut StdRng) -> usize {
        let mask = legal_mask(state, mover);
        match self {
            Agent::Random => {
                let legal: Vec<usize> = (0..NUM_CARDS).filter(|&j| mask[j] != 0.0).collect();
                legal[rng.gen_range(0..legal.len())]
            }
            Agent::Net(net) => {
                let v = encode(state, mover);
                let x = Tensor::from_slice(&v)
                    .reshape([1, INPUT_SIZE as i64])
                    .to_device(net.device());
                let (logits, _) = net.forward(&x);
                let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
                let mut buf = [0f32; NUM_CARDS];
                logits.copy_data(&mut buf, NUM_CARDS);
                // argmax over legal
                let mut best = usize::MAX;
                let mut best_v = f32::NEG_INFINITY;
                for j in 0..NUM_CARDS {
                    if mask[j] != 0.0 && buf[j] > best_v {
                        best_v = buf[j];
                        best = j;
                    }
                }
                best
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

fn snapshot_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

/// Recent prior snapshots (by sorted filename), excluding the candidate.
fn pool_snapshots(run_dir: &str, candidate_stem: &str, pool_size: usize) -> Vec<String> {
    let dir = Path::new(run_dir).join("snapshots");
    let mut snaps: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
                .map(|p| p.to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    snaps.sort(); // snap_hHHHHH_YYYYmmddTHHMMZ sorts chronologically
    snaps
        .into_iter()
        .filter(|p| snapshot_stem(p) != candidate_stem)
        .rev()
        .take(pool_size)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = flag(&args, "--run-dir", "runs/run1");
    let candidate = flag(&args, "--candidate", "");
    let games: usize = flag(&args, "--games", "200").parse().unwrap();
    let pool_size: usize = flag(&args, "--pool-size", "6").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    assert!(!candidate.is_empty(), "--candidate required");

    let dev = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let cand_stem = snapshot_stem(&candidate);
    let cand = Agent::Net(Net::load(&candidate, dev, Kind::Float));

    // Opponent pool: random + recent prior snapshots.
    let mut opponents: Vec<(String, Agent)> = vec![("random".to_string(), Agent::Random)];
    for snap in pool_snapshots(&run_dir, &cand_stem, pool_size) {
        let name = snapshot_stem(&snap);
        opponents.push((name, Agent::Net(Net::load(&snap, dev, Kind::Float))));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = String::from("[");
    for (oi, (name, opp)) in opponents.iter().enumerate() {
        let mut wins = 0usize;
        for g in 0..games {
            // Mirror seating across games to reduce first-mover bias.
            let cand_is_human = g % 2 == 0;
            let (human, bot) = if cand_is_human {
                (&cand, opp)
            } else {
                (opp, &cand)
            };
            let winner = play_match(human, bot, &mut rng);
            let cand_won = (winner == Player::Human) == cand_is_human;
            if cand_won {
                wins += 1;
            }
        }
        eprintln!(
            "  vs {:>22}: {}/{} ({:.1}%)",
            name,
            wins,
            games,
            100.0 * wins as f64 / games as f64
        );
        if oi > 0 {
            results.push(',');
        }
        results.push_str(&format!(
            "{{\"opponent\":\"{name}\",\"games\":{games},\"wins\":{wins}}}"
        ));
    }
    results.push(']');
    println!("{{\"candidate\":\"{cand_stem}\",\"results\":{results}}}");
}
