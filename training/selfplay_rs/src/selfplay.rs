//! Search-free self-play cohort generation (Phase 4).
//!
//! Plays full matches-to-21 current-vs-current with the loaded net. At each
//! decision the net's policy logits (masked to legal, temperature-applied) are
//! sampled for the move. Every decision is recorded; when a match ends, each
//! decision gets z = +1/-1 from that seat's frame (match win/loss). Rows are
//! written to a flat f32 cohort file the Python REINFORCE trainer consumes.
//!
//! Row layout (ROW_FLOATS = INPUT_SIZE + NUM_CARDS + 2):
//!   [ state(INPUT_SIZE) | legal_mask(NUM_CARDS) | action_index | z ]
//! File: u32 num_rows (LE), u32 row_floats (LE), then num_rows*row_floats f32 LE.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{encode, legal_mask, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::{Phase, Player, State, NUM_CARDS};

use crate::net::Net;

pub const ROW_FLOATS: usize = INPUT_SIZE + NUM_CARDS + 2;

pub struct Config {
    pub weights: String,
    pub out: String,
    pub matches: usize,
    pub batch: usize,
    pub temperature: f64,
    pub seed: u64,
    pub cpu: bool,
}

struct Decision {
    state: Vec<f32>,
    mask: [f32; NUM_CARDS],
    action: u32,
    seat: Player,
}

struct Game {
    state: State,
    decisions: Vec<Decision>,
}

impl Game {
    fn new(rng: &mut StdRng) -> Game {
        Game {
            state: State::new_match(rng),
            decisions: Vec::new(),
        }
    }
}

/// Temperature softmax over legal logits, then sample a canonical card index.
pub(crate) fn sample_action(logits: &[f32], mask: &[f32; NUM_CARDS], temp: f64, rng: &mut StdRng) -> usize {
    let legal: Vec<usize> = (0..NUM_CARDS).filter(|&j| mask[j] != 0.0).collect();
    let t = temp.max(1e-6) as f32;
    let maxl = legal
        .iter()
        .map(|&j| logits[j])
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = legal.iter().map(|&j| ((logits[j] - maxl) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (k, &j) in legal.iter().enumerate() {
        acc += exps[k];
        if r <= acc {
            return j;
        }
    }
    *legal.last().expect("no legal moves")
}

pub fn run(cfg: Config) {
    let dev = if cfg.cpu {
        Device::Cpu
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        eprintln!("warning: CUDA unavailable, self-play on CPU");
        Device::Cpu
    };
    let net = Net::load(&cfg.weights, dev, Kind::Float);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let batch = cfg.batch.min(cfg.matches).max(1);

    let mut slots: Vec<Option<Game>> = Vec::with_capacity(batch);
    let mut started = 0usize;
    for _ in 0..batch {
        slots.push(Some(Game::new(&mut rng)));
        started += 1;
    }

    let mut rows: Vec<f32> = Vec::new();
    let mut finished = 0usize;
    let mut wins = [0u64; 2];
    let mut total_decisions = 0u64;
    let start = Instant::now();

    while slots.iter().any(|s| s.is_some()) {
        let mut batch_idx: Vec<usize> = Vec::new();
        let mut enc_flat: Vec<f32> = Vec::new();

        // 1. Drive every slot to a Playing decision (finalizing / refilling at match end).
        for i in 0..batch {
            loop {
                let phase = match &slots[i] {
                    Some(g) => g.state.phase,
                    None => break,
                };
                match phase {
                    Phase::Playing => break,
                    Phase::TrickComplete => slots[i].as_mut().unwrap().state.advance_after_trick(),
                    Phase::RoundOver => slots[i].as_mut().unwrap().state.end_round(&mut rng),
                    Phase::MatchOver => {
                        let winner = slots[i].as_ref().unwrap().state.match_winner().unwrap();
                        wins[winner.idx()] += 1;
                        {
                            let g = slots[i].as_ref().unwrap();
                            for d in &g.decisions {
                                let z = if d.seat == winner { 1.0 } else { -1.0 };
                                rows.extend_from_slice(&d.state);
                                rows.extend_from_slice(&d.mask);
                                rows.push(d.action as f32);
                                rows.push(z);
                            }
                            total_decisions += g.decisions.len() as u64;
                        }
                        finished += 1;
                        if started < cfg.matches {
                            slots[i] = Some(Game::new(&mut rng));
                            started += 1;
                        } else {
                            slots[i] = None;
                            break;
                        }
                    }
                }
            }
            if let Some(g) = &slots[i] {
                if g.state.phase == Phase::Playing {
                    let mover = g.state.awaiting.unwrap();
                    enc_flat.extend_from_slice(&encode(&g.state, mover));
                    batch_idx.push(i);
                }
            }
        }
        if batch_idx.is_empty() {
            continue;
        }

        // 2. One batched forward over all pending decisions.
        let m = batch_idx.len() as i64;
        let x = Tensor::from_slice(&enc_flat)
            .reshape([m, INPUT_SIZE as i64])
            .to_device(dev);
        let (logits, _v) = net.forward(&x);
        let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
        let mut lbuf = vec![0f32; (m as usize) * NUM_CARDS];
        let ln = lbuf.len();
        logits.copy_data(&mut lbuf, ln);

        // 3. Sample + apply one move per pending game; record the decision.
        for (k, &i) in batch_idx.iter().enumerate() {
            let g = slots[i].as_mut().unwrap();
            let mover = g.state.awaiting.unwrap();
            let mask = legal_mask(&g.state, mover);
            let logit_row = &lbuf[k * NUM_CARDS..(k + 1) * NUM_CARDS];
            let action = sample_action(logit_row, &mask, cfg.temperature, &mut rng);
            let state_vec = enc_flat[k * INPUT_SIZE..(k + 1) * INPUT_SIZE].to_vec();
            g.decisions.push(Decision {
                state: state_vec,
                mask,
                action: action as u32,
                seat: mover,
            });
            let card = real_card_from_canon_index(action, g.state.trump.suit);
            g.state.apply(card);
        }
    }

    let n_rows = total_decisions as usize;
    write_cohort(&cfg.out, &rows, n_rows);

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "self-play: {} matches, {} rows ({:.1}/match), wins[H,B]={:?}, {:.1}s, {:.0} matches/s, {:.0} rows/s",
        finished,
        n_rows,
        n_rows as f64 / finished.max(1) as f64,
        wins,
        secs,
        finished as f64 / secs,
        n_rows as f64 / secs,
    );
}

pub(crate) fn write_cohort(path: &str, rows: &[f32], n_rows: usize) {
    assert_eq!(rows.len(), n_rows * ROW_FLOATS, "row buffer size mismatch");
    let f = File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    let mut w = BufWriter::new(f);
    w.write_all(&(n_rows as u32).to_le_bytes()).unwrap();
    w.write_all(&(ROW_FLOATS as u32).to_le_bytes()).unwrap();
    let mut bytes = Vec::with_capacity(rows.len() * 4);
    for &x in rows {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&bytes).unwrap();
    w.flush().unwrap();
}
