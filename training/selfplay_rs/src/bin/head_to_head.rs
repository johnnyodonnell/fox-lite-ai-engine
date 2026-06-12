//! head_to_head — play two snapshots against each other directly, cross-arch.
//!
//! Each side's arch (v1 pre-history MLP / v2 flat-token transformer / v3
//! trick-token transformer) is detected from its safetensors keys, and each
//! side encodes the shared game state with its own encoding (v1: 230, v2: 205,
//! v3: 209). Play mechanics match evaluate_rs:
//! greedy argmax over legal moves, mirrored seating, no draws. Results are
//! printed only — pool.json is untouched (the two runs' Elo scales are not
//! comparable anyway).
//!
//!   head_to_head --a runs/run4/snapshots/snap_x.safetensors \
//!                --b runs/run1/snapshots/snap_y.safetensors \
//!                [--games 200] [--seed 0]

use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{
    encode, encode_v1, encode_v2, legal_mask, real_card_from_canon_index, INPUT_SIZE,
    INPUT_SIZE_V1, INPUT_SIZE_V2,
};
use foxlite_core::{Phase, Player, State, NUM_CARDS};
use selfplay_rs::net::AnyNet;

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

struct NetAgent {
    net: AnyNet,
}

impl NetAgent {
    fn arch(&self) -> &'static str {
        match &self.net {
            AnyNet::V1(_) => "v1",
            AnyNet::V2(_) => "v2",
            AnyNet::V3(n) => n.flavor(),
        }
    }

    /// Greedy argmax over legal canonical card slots.
    fn act(&self, state: &State, mover: Player) -> usize {
        let mask = legal_mask(state, mover);
        let (v, size) = match self.net {
            AnyNet::V1(_) => (encode_v1(state, mover), INPUT_SIZE_V1),
            AnyNet::V2(_) => (encode_v2(state, mover), INPUT_SIZE_V2),
            AnyNet::V3(_) => (encode(state, mover), INPUT_SIZE),
        };
        let x = Tensor::from_slice(&v)
            .reshape([1, size as i64])
            .to_device(self.net.device());
        let (logits, _) = self.net.forward(&x);
        let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
        let mut buf = [0f32; NUM_CARDS];
        logits.copy_data(&mut buf, NUM_CARDS);
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

fn play_match(human: &NetAgent, bot: &NetAgent, rng: &mut StdRng) -> Player {
    let mut s = State::new_match(rng);
    loop {
        match s.phase {
            Phase::Playing => {
                let mover = s.awaiting.unwrap();
                let agent = if mover == Player::Human { human } else { bot };
                let action = agent.act(&s, mover);
                let card = real_card_from_canon_index(action, s.trump.suit);
                s.apply(card);
            }
            Phase::TrickComplete => s.advance_after_trick(),
            Phase::RoundOver => s.end_round(rng),
            Phase::MatchOver => return s.match_winner().unwrap(),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a_path = flag(&args, "--a", "");
    let b_path = flag(&args, "--b", "");
    let games: usize = flag(&args, "--games", "200").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    assert!(!a_path.is_empty() && !b_path.is_empty(), "--a and --b required");

    let dev = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let a = NetAgent { net: AnyNet::load_auto(&a_path, dev, Kind::Float) };
    let b = NetAgent { net: AnyNet::load_auto(&b_path, dev, Kind::Float) };
    let stem = |p: &str| {
        std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| p.to_string())
    };
    let (a_name, b_name) = (stem(&a_path), stem(&b_path));
    println!("[h2h] a={a_name} ({}) b={b_name} ({}) games={games} device={dev:?}", a.arch(), b.arch());

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a_wins = 0usize;
    let mut a_wins_as_human = 0usize;
    let mut a_wins_as_bot = 0usize;
    for g in 0..games {
        let a_is_human = g % 2 == 0;
        let (human, bot) = if a_is_human { (&a, &b) } else { (&b, &a) };
        let winner = play_match(human, bot, &mut rng);
        if (winner == Player::Human) == a_is_human {
            a_wins += 1;
            if a_is_human {
                a_wins_as_human += 1;
            } else {
                a_wins_as_bot += 1;
            }
        }
        if (g + 1) % 50 == 0 {
            println!("[h2h] {}/{games}: {a_name} {a_wins} - {} {b_name}", g + 1, g + 1 - a_wins);
        }
    }

    let p = a_wins as f64 / games as f64;
    println!("[h2h] result: {a_name} {a_wins} - {} {b_name} ({:.1}%)", games - a_wins, 100.0 * p);
    println!(
        "[h2h] {a_name} seat split: {a_wins_as_human}/{} as first-seat, {a_wins_as_bot}/{} as second-seat",
        games / 2 + games % 2,
        games / 2
    );
    if p > 0.0 && p < 1.0 {
        let diff = 400.0 * (p / (1.0 - p)).log10();
        // 95% CI on the win rate -> Elo diff interval
        let se = (p * (1.0 - p) / games as f64).sqrt();
        let (lo, hi) = ((p - 1.96 * se).max(1e-9), (p + 1.96 * se).min(1.0 - 1e-9));
        let lo_d = 400.0 * (lo / (1.0 - lo)).log10();
        let hi_d = 400.0 * (hi / (1.0 - hi)).log10();
        println!("[h2h] elo diff ({a_name} - {b_name}): {diff:+.0} [95% CI {lo_d:+.0}, {hi_d:+.0}]");
    } else {
        println!("[h2h] elo diff: sweep — outside Elo scale");
    }
}
