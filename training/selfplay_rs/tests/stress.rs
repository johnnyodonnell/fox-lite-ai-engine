//! Pipeline liveness stress test: many micro-cohorts with a stub policy, sized
//! so every cohort lives in the tail ramp-down — partial batches, bursts of
//! near-simultaneous completions — where both historical deadlocks bit
//! (eec2d602: lost wakeup; 46bb17d: stale in_flight snapshot). Each cohort runs
//! under a watchdog, so a wedge fails the test with the offending config
//! instead of hanging forever.
//!
//! Empirical, not exhaustive: it samples OS-scheduled interleavings, weighted
//! toward the dangerous region (loom would be the exhaustive complement).
//! `STRESS_ITERS` overrides the iteration count; runs on CPU, no weights.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tch::{Device, Kind, Tensor};

use foxlite_core::NUM_CARDS;
use selfplay_rs::pipeline::{run_with_forward, Config};

/// Cohorts take ~50ms; a cohort this late is wedged, not slow.
const COHORT_TIMEOUT: Duration = Duration::from_secs(60);

/// Uniform-policy stub: zero logits (random legal play), with occasional short
/// sleeps so inference randomly falls behind or runs ahead of the workers —
/// each cohort tail then drains from a different queue/in-flight split.
fn stub_forward(x: &Tensor) -> Tensor {
    let mut rng = rand::thread_rng();
    if rng.gen_ratio(1, 4) {
        thread::sleep(Duration::from_micros(rng.gen_range(0..200)));
    }
    Tensor::zeros([x.size()[0], NUM_CARDS as i64], (Kind::Float, Device::Cpu))
}

#[test]
fn micro_cohorts_never_wedge() {
    let iters: usize = std::env::var("STRESS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let out = std::env::temp_dir().join(format!("stress_cohort_{}.bin", std::process::id()));
    let mut rng = StdRng::seed_from_u64(0xF0E1_D2C3);

    for i in 0..iters {
        // Small batches with matches a few multiples above them: every cohort
        // ends in partial batches with completions bunched across many workers.
        let batch = rng.gen_range(1..=8usize);
        let matches = batch + rng.gen_range(1..=2 * batch + 2);
        let concurrency = rng.gen_range(batch..=2 * batch + 2);
        let threads = [1, 2, 4, 8, 16][rng.gen_range(0..5)];
        let slots = rng.gen_range(1..=3usize);
        let desc = format!(
            "iter {i}: matches={matches} batch={batch} conc={concurrency} threads={threads} slots={slots}"
        );

        let cfg = Config {
            weights: String::new(), // unused: the forward is injected
            out: out.to_str().unwrap().to_string(),
            matches,
            batch,
            concurrency,
            n_threads: threads,
            slots,
            temperature: 1.0,
            temp_end: 1.0,
            seed: i as u64,
            cpu: true,
        };

        // Watchdog: run the cohort on a helper thread; a wedge leaks that thread
        // and fails the test, which beats waiting on it forever.
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(run_with_forward(&cfg, &stub_forward, Device::Cpu));
        });
        match rx.recv_timeout(COHORT_TIMEOUT) {
            Ok(finished) => assert_eq!(finished, matches, "lost matches ({desc})"),
            Err(_) => panic!("pipeline wedged ({desc}); no ack within {COHORT_TIMEOUT:?}"),
        }

        if (i + 1) % 100 == 0 {
            eprintln!("[stress] {}/{} cohorts ok", i + 1, iters);
        }
    }
    let _ = std::fs::remove_file(&out);
}
