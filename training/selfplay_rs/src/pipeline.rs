//! Parallel self-play cohort generation: N game-worker threads + a single
//! batched-inference thread, decoupled by two queues.
//!
//! The baseline `selfplay::run` is single-threaded: it drives every game's
//! rules logic (encode / sample / apply) serially on one CPU thread between
//! batched GPU forwards, so the GPU sits ~70% idle (CPU-bound). This module
//! parallelizes the rules logic across `n_threads` workers and overlaps it with
//! the GPU forward, keeping the device fed.
//!
//! Topology (search-free fox-lite — each decision needs exactly ONE forward):
//!   - workers own no torch; they advance a game's FSM until it yields ONE
//!     decision to evaluate (encode the state, push to the pre-inference queue)
//!     or the match ends (finalize rows -> sink). An empty ready queue with
//!     budget left => spawn a fresh game (concurrency self-regulates);
//!   - ONE pre-inference queue: workers push staged encodings, the inference
//!     thread pops up to BATCH at a time. All queue/counter bookkeeping lives
//!     under a single mutex (see `PipelineState`);
//!   - ONE inference thread: gathers a batch off the pre-inference queue, runs
//!     the net forward, copies logits back to the host, attaches each game's
//!     logit row as its `pending` result, and requeues the games on the ready
//!     queue for a worker to sample + apply.
//!
//! Output is byte-identical in layout to `selfplay::run` (same `write_cohort`),
//! so the Python trainer consumes it unchanged.

use std::collections::VecDeque;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{encode, legal_mask, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::{Phase, Player, State, NUM_CARDS};

use crate::cuda_event::CudaEvent;
use crate::net::Net;
use crate::selfplay::{
    emit_decision, round_decides_match, round_points, round_reward, round_temp, sample_action,
    write_cohort,
};

pub struct Config {
    pub weights: String,
    pub out: String,
    pub matches: usize,
    pub batch: usize,       // GPU forward width (max rows per forward)
    pub concurrency: usize, // games kept in flight; > batch overlaps CPU with GPU
    pub n_threads: usize,   // game-worker threads
    pub slots: usize,       // forwards in flight (inference can run ahead of scatter)
    pub temperature: f64,   // sampling temperature at a round's first trick
    pub temp_end: f64,      // ...annealed to this by the round's last trick
    pub seed: u64,
    pub cpu: bool,
}

/// One recorded decision. Non-deciding rounds are flushed to `InFlight::rows`
/// scored on their own points; the match-deciding round's decisions stay here
/// and are flushed with z = ±1 when the match ends.
struct Decision {
    state: Vec<f32>,
    mask: [f32; NUM_CARDS],
    action: u32,
    seat: Player,
}

/// A game in flight through the pipeline.
struct InFlight {
    state: State,
    decisions: Vec<Decision>, // current round only (drained at each round boundary)
    rows: Vec<f32>,           // cohort rows for completed non-deciding rounds
    n_rows: u64,              // decisions already emitted into `rows`
    rng: StdRng,
    staged_enc: Vec<f32>, // encoding of the decision currently awaiting a forward
    staged_mover: Player, // seat to move at that decision
    // Forward result: the whole batch's logits (shared, one alloc per forward)
    // plus this game's row in it. The worker slices its own row when sampling —
    // no per-game copy on the serial inference thread.
    pending: Option<(Arc<Vec<f32>>, usize)>,
}

/// Per-worker tallies, merged once at join (keeps the cohort rows and counters
/// off the shared ready-lock — only the in-flight bookkeeping is shared).
#[derive(Default)]
struct WorkerOut {
    rows: Vec<f32>,
    finished: usize,
    wins: [u64; 2],
    total_decisions: u64,
}

// --- per-game seeding (splitmix64; just a deterministic spread per game id) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// All pipeline bookkeeping — both queues, the spawn/finish counters, and the
/// shutdown flag — under ONE mutex. Every condvar predicate in this module
/// reads only fields of this struct while holding this mutex; keep it that
/// way. Both deadlocks this pipeline has shipped (eec2d602: lost wakeup;
/// 46bb17d: stale count snapshot) were races between a wait predicate and
/// predicate state maintained across a second lock — a class this layout
/// makes unrepresentable.
///
/// Queue sizes are bounded by construction, not by backpressure: every live
/// game is in at most one queue, so each queue holds <= `in_flight` <=
/// concurrency games.
struct PipelineState {
    ready: VecDeque<Box<InFlight>>,    // forward result attached; awaiting a worker
    preinfer: VecDeque<Box<InFlight>>, // staged encoding; awaiting the inference thread
    started: usize,                    // games spawned so far (<= matches)
    in_flight: usize,                  // games alive (spawned, not yet finalized)
    closed: bool,                      // cohort complete: drain and exit
}

struct Shared {
    state: Mutex<PipelineState>,
    worker_cv: Condvar, // workers: a requeued game arrived, or closed
    gather_cv: Condvar, // inference: full batch / tail partial batch / closed
    matches: usize,
    concurrency: usize,
    temperature: f64,
    temp_end: f64,
    seed: u64,
}

/// Queue a staged game for the inference thread. Never blocks: the queue only
/// holds live games, so it is already bounded by `concurrency` (and `closed`
/// cannot be set while the caller holds a live game — in_flight > 0).
fn stage(shared: &Shared, g: Box<InFlight>) {
    let mut s = shared.state.lock().unwrap();
    s.preinfer.push_back(g);
    drop(s);
    shared.gather_cv.notify_one();
}

/// Inference gather: block until either a full `batch` is staged, or every
/// remaining in-flight game is staged (ramp-down / tail: nothing more can
/// arrive, so take the partial batch — `in_flight <= preinfer.len()` covers
/// the moment the last stragglers land), or the cohort is complete. Returns
/// None only on completion.
fn gather(shared: &Shared, batch: usize) -> Option<Vec<Box<InFlight>>> {
    let mut s = shared.state.lock().unwrap();
    loop {
        if s.closed {
            return None;
        }
        if s.preinfer.len() >= batch || (s.preinfer.len() >= s.in_flight && !s.preinfer.is_empty()) {
            let take = batch.min(s.preinfer.len());
            return Some(s.preinfer.drain(..take).collect());
        }
        s = shared.gather_cv.wait(s).unwrap();
    }
}

fn spawn_fresh(base_seed: u64, id: u64) -> Box<InFlight> {
    let mut rng = StdRng::seed_from_u64(mix64(base_seed ^ mix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15))));
    let state = State::new_match(&mut rng);
    Box::new(InFlight {
        state,
        decisions: Vec::new(),
        rows: Vec::new(),
        n_rows: 0,
        rng,
        staged_enc: Vec::new(),
        staged_mover: Player::Human,
        pending: None,
    })
}

enum Advance {
    Eval,
    Done(Player), // match winner
}

/// Drive `g` from its current phase to the next Playing decision (staging its
/// encoding) or to MatchOver. Mirrors the slot-advance logic in `selfplay::run`.
fn advance(g: &mut InFlight) -> Advance {
    loop {
        match g.state.phase {
            Phase::Playing => {
                let mover = g.state.awaiting.unwrap();
                g.staged_enc = encode(&g.state, mover);
                g.staged_mover = mover;
                return Advance::Eval;
            }
            Phase::TrickComplete => g.state.advance_after_trick(),
            Phase::RoundOver => {
                let pts = round_points(&g.state);
                // Non-deciding round: score each decision on this round's points
                // and move it to `rows`, leaving `decisions` empty for the next
                // round. A deciding round is left for `finalize_local` (match z).
                if !round_decides_match(&g.state, pts) {
                    let z = round_reward(pts);
                    let decisions = std::mem::take(&mut g.decisions);
                    g.n_rows += decisions.len() as u64;
                    for d in &decisions {
                        emit_decision(&mut g.rows, &d.state, &d.mask, d.action, z[d.seat.idx()]);
                    }
                }
                let mut rng = std::mem::replace(&mut g.rng, StdRng::seed_from_u64(0));
                g.state.end_round(&mut rng);
                g.rng = rng;
            }
            Phase::MatchOver => return Advance::Done(g.state.match_winner().unwrap()),
        }
    }
}

/// Apply the forward result for the staged decision: sample a move, record the
/// decision, and play it. Leaves `g` at the post-move phase for `advance`.
fn apply_pending(g: &mut InFlight, temp_start: f64, temp_end: f64) {
    let (logits, row) = g.pending.take().expect("apply_pending without pending result");
    let mover = g.staged_mover;
    let mask = legal_mask(&g.state, mover);
    let logit_row = &logits[row * NUM_CARDS..(row + 1) * NUM_CARDS];
    let temp = round_temp(temp_start, temp_end, g.state.trick_num);
    let action = sample_action(logit_row, &mask, temp, &mut g.rng);
    let state_vec = std::mem::take(&mut g.staged_enc);
    g.decisions.push(Decision { state: state_vec, mask, action: action as u32, seat: mover });
    let card = real_card_from_canon_index(action, g.state.trump.suit);
    g.state.apply(card);
}

/// Pop a game to work on: an existing requeued game, a freshly spawned one if
/// there's spawn budget and concurrency headroom, or None when the cohort is
/// complete (all matches finished). Blocks while games are out at inference.
fn acquire(shared: &Shared) -> Option<Box<InFlight>> {
    let mut s = shared.state.lock().unwrap();
    loop {
        if s.closed {
            return None; // cohort complete
        }
        if let Some(g) = s.ready.pop_front() {
            return Some(g);
        }
        if s.started < shared.matches && s.in_flight < shared.concurrency {
            let id = s.started as u64;
            s.started += 1;
            s.in_flight += 1;
            return Some(spawn_fresh(shared.seed, id));
        }
        s = shared.worker_cv.wait(s).unwrap();
    }
}

/// Flush a finished game's decisions into the worker-local cohort buffer.
fn finalize_local(out: &mut WorkerOut, g: &InFlight, winner: Player) {
    // Non-deciding rounds were already scored on their own points into `g.rows`.
    out.rows.extend_from_slice(&g.rows);
    out.total_decisions += g.n_rows;
    // Whatever decisions remain are the match-deciding round; score them on the
    // match outcome, not the points that round scored.
    for d in &g.decisions {
        let z = if d.seat == winner { 1.0f32 } else { -1.0f32 };
        emit_decision(&mut out.rows, &d.state, &d.mask, d.action, z);
    }
    out.total_decisions += g.decisions.len() as u64;
    out.wins[winner.idx()] += 1;
    out.finished += 1;
}

/// Shared in-flight bookkeeping for one completed game (no row copy here).
fn complete_one(shared: &Shared) {
    let mut s = shared.state.lock().unwrap();
    s.in_flight -= 1;
    if s.started >= shared.matches && s.in_flight == 0 {
        // Cohort complete: wake everyone to drain and shut down cleanly.
        s.closed = true;
        drop(s);
        shared.worker_cv.notify_all();
        shared.gather_cv.notify_all();
    } else {
        drop(s);
        // in_flight shrank, so gather's tail predicate may now fire.
        shared.gather_cv.notify_one();
    }
}

fn worker_loop(shared: Arc<Shared>) -> WorkerOut {
    let mut out = WorkerOut::default();
    while let Some(mut g) = acquire(&shared) {
        if g.pending.is_some() {
            apply_pending(&mut g, shared.temperature, shared.temp_end);
        }
        match advance(&mut g) {
            Advance::Eval => stage(&shared, g),
            Advance::Done(winner) => {
                finalize_local(&mut out, &g, winner);
                complete_one(&shared);
            }
        }
    }
    out
}

/// One in-flight forward handed from the inference thread to the scatter thread.
/// `logits` is still on the GPU; the scatter thread reads it back (D2H) after the
/// `event` fires, so the readback overlaps the inference thread's next forward.
struct WorkItem {
    games: Vec<Box<InFlight>>,
    logits: Tensor, // GPU [m, NUM_CARDS]
    event: CudaEvent,
}

/// Inference (GPU producer): gather a batch off the pre-inference queue, copy it
/// to the device, launch the forward, record a completion event, and hand the
/// in-flight forward to the scatter thread — then immediately loop to launch the
/// next batch. The readback (the old blocking `to_device(Cpu)`) now lives on the
/// scatter thread, so this thread never blocks on the GPU and keeps it fed.
fn inference_loop(
    shared: Arc<Shared>,
    forward: &dyn Fn(&Tensor) -> Tensor,
    dev: Device,
    batch: usize,
    work_tx: SyncSender<WorkItem>,
) {
    let mut enc_flat: Vec<f32> = Vec::with_capacity(batch * INPUT_SIZE);
    loop {
        let games = match gather(&shared, batch) {
            Some(g) => g,
            None => return, // cohort complete: dropping work_tx drains the scatter thread
        };
        let m = games.len();
        enc_flat.clear();
        for g in &games {
            enc_flat.extend_from_slice(&g.staged_enc);
        }
        let x = Tensor::from_slice(&enc_flat)
            .reshape([m as i64, INPUT_SIZE as i64])
            .to_device(dev);
        let logits = forward(&x); // async on the stream
        let logits = logits.to_kind(Kind::Float); // stays on GPU; scatter reads it back
        let event = CudaEvent::new();
        event.record(); // fires when this forward completes
        if work_tx.send(WorkItem { games, logits, event }).is_err() {
            return; // scatter gone
        }
    }
}

/// Scatter (GPU consumer): wait each forward's event, read its logits back to the
/// host, attach each game's (shared logits, row), and requeue it for a worker.
/// Overlaps the inference thread's next forward. Exits when the inference thread
/// drops `work_tx` (shutdown) and the channel drains.
fn scatter_loop(shared: Arc<Shared>, work_rx: Receiver<WorkItem>) {
    while let Ok(item) = work_rx.recv() {
        item.event.sync(); // wait the forward
        let m = item.games.len();
        let logits = item.logits.to_device(Device::Cpu).contiguous();
        // One host buffer per forward, shared by Arc; workers slice their own row.
        let mut host = vec![0f32; m * NUM_CARDS];
        logits.copy_data(&mut host, m * NUM_CARDS);
        let host = Arc::new(host);

        // Attach each game's (shared logits, row) and requeue for a worker.
        let mut s = shared.state.lock().unwrap();
        for (k, mut g) in item.games.into_iter().enumerate() {
            g.pending = Some((Arc::clone(&host), k));
            s.ready.push_back(g);
        }
        drop(s);
        shared.worker_cv.notify_all();
    }
}

/// Pick the self-play device (CUDA if available, unless `--cpu`).
pub fn pick_device(cpu: bool) -> Device {
    if cpu {
        Device::Cpu
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        eprintln!("warning: CUDA unavailable, self-play on CPU");
        Device::Cpu
    }
}

pub fn run(cfg: Config) {
    let dev = pick_device(cfg.cpu);
    let net = Net::load(&cfg.weights, dev, Kind::Float);
    run_with_net(&cfg, &net, dev);
}

/// Persistent worker: initialize CUDA/libtorch once, then run one cohort per
/// newline-delimited JSON command on stdin (`{weights,out,matches,batch,
/// concurrency,threads,slots,temperature,seed}`, optional `temp_end`/`quit`). Reloads
/// weights per command (a few ms for this net) and acks each cohort with a
/// `{"done":true}` line — the cohort itself is handed off via the `out` file.
pub fn serve(dev: Device) {
    use std::io::{BufRead, Write};

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    println!("{}", serde_json::json!({"ready": true}));
    stdout.flush().expect("flush ready");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin closed -> shut down
        };
        if line.trim().is_empty() {
            continue;
        }
        let cmd: ServeCmd = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("bad serve command {line:?}: {e}"));
        if cmd.quit {
            break;
        }
        let cfg = cmd.into_config();
        // Reload weights each cohort: the orchestrator rewrites the same path
        // (serving_weights.safetensors) every iteration, so a path cache would miss.
        let net = Net::load(&cfg.weights, dev, Kind::Float);
        run_with_net(&cfg, &net, dev);
        println!("{}", serde_json::json!({"done": true}));
        stdout.flush().expect("flush done");
    }
}

/// One self-play command from the orchestrator (mirrors `Config`'s knobs).
#[derive(serde::Deserialize)]
struct ServeCmd {
    weights: String,
    out: String,
    matches: usize,
    batch: usize,
    concurrency: usize,
    threads: usize,
    slots: usize,
    temperature: f64,
    // Optional so older callers (and the `quit` message) still parse; defaults to
    // `temperature` in `into_config`, i.e. no annealing unless explicitly set.
    #[serde(default)]
    temp_end: Option<f64>,
    seed: u64,
    #[serde(default)]
    cpu: bool,
    #[serde(default)]
    quit: bool,
}

impl ServeCmd {
    fn into_config(self) -> Config {
        Config {
            weights: self.weights,
            out: self.out,
            matches: self.matches,
            batch: self.batch,
            concurrency: self.concurrency,
            n_threads: self.threads,
            slots: self.slots,
            temperature: self.temperature,
            temp_end: self.temp_end.unwrap_or(self.temperature),
            seed: self.seed,
            cpu: self.cpu,
        }
    }
}

/// Generate one cohort with an already-loaded net (shared by `run` and `serve`).
fn run_with_net(cfg: &Config, net: &Net, dev: Device) {
    run_with_forward(cfg, &|x| net.forward(x).0, dev);
}

/// Generate one cohort with an injected policy forward ([m, INPUT_SIZE]
/// encodings -> [m, NUM_CARDS] logits). Production wraps the real `Net`; the
/// liveness stress test (tests/stress.rs) injects a stub so the pipeline's
/// synchronization can be hammered without weights or a GPU. Returns the
/// number of finished matches.
pub fn run_with_forward(cfg: &Config, forward: &dyn Fn(&Tensor) -> Tensor, dev: Device) -> usize {
    let batch = cfg.batch.max(1).min(cfg.matches.max(1));
    let concurrency = cfg.concurrency.max(batch).min(cfg.matches.max(1));
    let n_threads = cfg.n_threads.max(1);
    let slots = cfg.slots.max(1);

    let shared = Arc::new(Shared {
        state: Mutex::new(PipelineState {
            ready: VecDeque::new(),
            preinfer: VecDeque::new(),
            started: 0,
            in_flight: 0,
            closed: false,
        }),
        worker_cv: Condvar::new(),
        gather_cv: Condvar::new(),
        matches: cfg.matches,
        concurrency,
        temperature: cfg.temperature,
        temp_end: cfg.temp_end,
        seed: cfg.seed,
    });

    let start = Instant::now();
    let workers: Vec<_> = (0..n_threads)
        .map(|_| {
            let sh = shared.clone();
            thread::spawn(move || worker_loop(sh))
        })
        .collect();

    // The scatter thread reads each forward's logits back to the host and requeues
    // the games; bound the channel at `slots` so the inference thread can run at
    // most `slots` forwards ahead of the readback (backpressure).
    let (work_tx, work_rx) = sync_channel::<WorkItem>(slots);
    let scatter = {
        let sh = shared.clone();
        thread::spawn(move || scatter_loop(sh, work_rx))
    };

    // Inference runs on this thread (owns the net / GPU). On shutdown it returns
    // and drops `work_tx`, which drains and ends the scatter thread.
    inference_loop(shared.clone(), forward, dev, batch, work_tx);
    scatter.join().expect("scatter thread panicked");

    // Merge the per-worker cohort buffers (concatenated; row order is arbitrary,
    // which is fine — the trainer shuffles).
    let mut rows: Vec<f32> = Vec::new();
    let (mut finished, mut total_decisions, mut wins) = (0usize, 0u64, [0u64; 2]);
    for w in workers {
        let o = w.join().expect("worker panicked");
        rows.extend_from_slice(&o.rows);
        finished += o.finished;
        total_decisions += o.total_decisions;
        wins[0] += o.wins[0];
        wins[1] += o.wins[1];
    }

    let n_rows = total_decisions as usize;
    write_cohort(&cfg.out, &rows, n_rows);
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "self-play (pipeline): {} matches, {} rows ({:.1}/match), wins[H,B]={:?}, {} threads, batch {}, conc {}, slots {}, {:.1}s, {:.0} matches/s, {:.0} rows/s",
        finished,
        n_rows,
        n_rows as f64 / finished.max(1) as f64,
        wins,
        n_threads,
        batch,
        concurrency,
        slots,
        secs,
        finished as f64 / secs,
        n_rows as f64 / secs,
    );
    finished
}
