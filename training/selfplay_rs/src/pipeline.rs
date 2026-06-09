//! Continuous ISMCTS self-play pipeline (100% Rust, leaf-batched).
//!
//! Port of the chess-ai-engine pipeline topology onto Fox-Lite, with the heavy
//! inference stack (AOTI / CUDA-graph / bf16 / pinned slots) swapped for the
//! simple `tch` fp32 `Net::forward` — the Fox-Lite net is tiny, so a plain forward
//! keeps the GPU fed and the code small.
//!
//! Topology:
//!   - N worker threads each drive one game's ISMCTS FSM until it yields ONE leaf
//!     to evaluate (root expand or a simulation leaf) or the game finishes (rows
//!     streamed to the sink). An empty queue => spawn a fresh game (dynamic game
//!     count, self-regulating to fill the pipe);
//!   - ONE bounded ply-priority pre-inference queue (most real-moves-made first),
//!     so games closest to finishing are evaluated soonest;
//!   - ONE inference thread: gathers BATCH leaf encodings, runs the fp32 forward,
//!     records a CUDA event, hands the in-flight forward to the scatter thread,
//!     and loops to launch the next batch. It also mtime-polls the weights sidecar
//!     and `Net::reload`s in place between batches (weight hot-reload);
//!   - ONE scatter thread: waits the event, reads (logits, value) back to the
//!     host, attaches each game's result, and requeues it for a worker.
//!
//! Each ISMCTS decision re-determinizes the hidden cards every simulation (true
//! ISMCTS); the leaf-batching means a single game contributes ~`sims` forwards per
//! move, all interleaved with every other in-flight game's forwards.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{self, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use foxlite_core::determinize::determinize;
use foxlite_core::encode::{encode, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::mcts::{
    add_dirichlet_noise, backprop, expand_node, new_root, sample_move, temperature, visits_to_pi,
    walk_to_leaf, Node, WalkResult,
};
use foxlite_core::{Phase, Player, State, NUM_CARDS};

use crate::cuda_event::CudaEvent;
use crate::net::Net;

pub const ENC_LEN: usize = INPUT_SIZE; // 230
pub const POLICY_SIZE: usize = NUM_CARDS; // 33

/// One finished-game training row: (state[230], pi[33], z).
pub type Row = (Vec<f32>, [f32; NUM_CARDS], f32);

pub struct Config {
    pub sims: usize,
    pub add_root_noise: bool,
    pub seed: u64,
    pub n_threads: usize,
    pub n_slots: usize, // GPU forwards kept in flight (inference runs ahead of scatter)
    pub batch: usize,   // GPU forward width
    pub weights_path: String,
    pub reload_every: Duration,
    pub cpu: bool,
}

// --- per-game seeding (splitmix64; determinism not required, just spread) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game ISMCTS FSM ------------------------------------------------------
enum GamePhase {
    NeedRootExpand,
    Simulating(usize),
}
enum LeafKind {
    RootExpand,
    SimLeaf,
}
struct LeafCtx {
    kind: LeafKind,
    path: Vec<usize>,
    leaf_idx: usize,
    det: State,    // determinized state at the leaf (clone of the true state for the root)
    mover: Player, // seat to move at the leaf
}
struct EvalResult {
    logits: Arc<Vec<f32>>, // whole batch's policy [m * 33]; the worker slices its row
    row: usize,
    value: f32,
}
/// One recorded decision; flushed to a row with z = ±1 when the match ends.
struct Decision {
    state_enc: Vec<f32>,
    pi: [f32; NUM_CARDS],
    seat: Player,
}
struct InFlight {
    id: u64,
    state: State,     // the TRUE match state
    searcher: Player, // seat to move at the current decision (= state.awaiting)
    arena: Vec<Node>, // ISMCTS tree for the current decision
    rng: StdRng,
    decisions: Vec<Decision>,
    phase: GamePhase,
    pending: Option<EvalResult>,
    leaf_ctx: Option<LeafCtx>,
    staged_enc: Vec<f32>, // encoding of the leaf currently awaiting a forward
}

fn spawn_fresh(shared: &Shared) -> InFlight {
    let id = shared.id_counter.fetch_add(1, AtomicOrdering::Relaxed);
    let seed = mix64(shared.config.seed ^ mix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let mut rng = StdRng::seed_from_u64(seed);
    let state = State::new_match(&mut rng);
    let searcher = state.awaiting.expect("fresh match awaits a leader");
    InFlight {
        id,
        state,
        searcher,
        arena: new_root(searcher),
        rng,
        decisions: Vec::new(),
        phase: GamePhase::NeedRootExpand,
        pending: None,
        leaf_ctx: None,
        staged_enc: Vec::new(),
    }
}

fn finalize_rows(g: &mut InFlight) -> Vec<Row> {
    let winner = g.state.match_winner().expect("finalize before MatchOver");
    g.decisions
        .drain(..)
        .map(|d| {
            let z = if d.seat == winner { 1.0f32 } else { -1.0f32 };
            (d.state_enc, d.pi, z)
        })
        .collect()
}

/// Apply the forward result for the staged leaf: expand it and backprop (or, for
/// the root, expand + add noise). Mirrors the eval branch of `mcts::run_search`.
fn apply_result(g: &mut InFlight, res: EvalResult, config: &Config) {
    let ctx = g.leaf_ctx.take().expect("pending result without leaf_ctx");
    let logits = &res.logits[res.row * POLICY_SIZE..(res.row + 1) * POLICY_SIZE];
    let value = res.value as f64;
    match ctx.kind {
        LeafKind::RootExpand => {
            expand_node(&mut g.arena, 0, &ctx.det, logits, value, g.searcher);
            if config.add_root_noise {
                add_dirichlet_noise(&mut g.arena, 0, &mut g.rng);
            }
            g.phase = GamePhase::Simulating(0);
        }
        LeafKind::SimLeaf => {
            expand_node(&mut g.arena, ctx.leaf_idx, &ctx.det, logits, value, g.searcher);
            let v_ref = if ctx.mover == g.searcher { value } else { -value };
            backprop(&mut g.arena, &ctx.path, v_ref);
            if let GamePhase::Simulating(s) = g.phase {
                g.phase = GamePhase::Simulating(s + 1);
            }
        }
    }
}

/// Pick the real move from the finished search, record the decision, play it, and
/// advance the true state to the next decision. Returns true if the match ended.
fn step_move(g: &mut InFlight) -> bool {
    let temp = temperature(g.state.trick_num);
    let pi = visits_to_pi(&g.arena, 0, temp);
    let state_enc = encode(&g.state, g.searcher);
    g.decisions.push(Decision { state_enc, pi, seat: g.searcher });

    let mv = sample_move(&g.arena, 0, temp, &mut g.rng);
    let card = real_card_from_canon_index(mv, g.state.trump.suit);
    g.state.apply(card);
    loop {
        match g.state.phase {
            Phase::Playing => break,
            Phase::TrickComplete => g.state.advance_after_trick(),
            Phase::RoundOver => {
                let mut rng = std::mem::replace(&mut g.rng, StdRng::seed_from_u64(0));
                g.state.end_round(&mut rng);
                g.rng = rng;
            }
            Phase::MatchOver => break,
        }
    }
    if g.state.phase == Phase::MatchOver {
        return true;
    }
    g.searcher = g.state.awaiting.expect("next decision awaits a mover");
    g.arena = new_root(g.searcher);
    g.phase = GamePhase::NeedRootExpand;
    false
}

enum Advance {
    Eval,
    Done(Vec<Row>),
}

/// Drive the game until it stages one leaf for evaluation or finishes.
fn advance_until_eval_or_done(g: &mut InFlight, config: &Config) -> Advance {
    loop {
        match g.phase {
            GamePhase::NeedRootExpand => {
                g.staged_enc = encode(&g.state, g.searcher);
                g.leaf_ctx = Some(LeafCtx {
                    kind: LeafKind::RootExpand,
                    path: vec![0],
                    leaf_idx: 0,
                    det: g.state.clone(),
                    mover: g.searcher,
                });
                return Advance::Eval;
            }
            GamePhase::Simulating(s) => {
                if s >= config.sims {
                    if step_move(g) {
                        return Advance::Done(finalize_rows(g));
                    }
                    continue;
                }
                let mut det = determinize(&g.state, g.searcher, &mut g.rng);
                match walk_to_leaf(&mut g.arena, 0, &mut det, g.searcher) {
                    WalkResult::Terminal { path, v_ref } => {
                        backprop(&mut g.arena, &path, v_ref);
                        g.phase = GamePhase::Simulating(s + 1);
                    }
                    WalkResult::Eval { path, mover } => {
                        let leaf_idx = *path.last().unwrap();
                        g.staged_enc = encode(&det, mover);
                        g.leaf_ctx = Some(LeafCtx {
                            kind: LeafKind::SimLeaf,
                            path,
                            leaf_idx,
                            det,
                            mover,
                        });
                        return Advance::Eval;
                    }
                }
            }
        }
    }
}

// --- ply-priority queues (most real moves made first) ------------------------
struct Queued {
    key: u64,
    seq: u64,
    item: Box<InFlight>,
}
impl PartialEq for Queued {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.seq == o.seq
    }
}
impl Eq for Queued {}
impl Ord for Queued {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.cmp(&o.key).then_with(|| self.seq.cmp(&o.seq))
    }
}
impl PartialOrd for Queued {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
fn priority_key(g: &InFlight) -> u64 {
    g.decisions.len() as u64
}

/// Bounded, blocking ply-priority pre-inference queue (chess's PreInferQueue).
struct PreInferQueue {
    inner: Mutex<BinaryHeap<Queued>>,
    not_empty: Condvar,
    not_full: Condvar,
    cap: usize,
    closed: AtomicBool,
}
impl PreInferQueue {
    fn new(cap: usize) -> PreInferQueue {
        PreInferQueue {
            inner: Mutex::new(BinaryHeap::new()),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            cap,
            closed: AtomicBool::new(false),
        }
    }
    fn push(&self, g: Box<InFlight>) -> Result<(), ()> {
        let mut q = self.inner.lock().unwrap();
        while q.len() >= self.cap && !self.closed.load(AtomicOrdering::Relaxed) {
            q = self.not_full.wait(q).unwrap();
        }
        if self.closed.load(AtomicOrdering::Relaxed) {
            return Err(());
        }
        let key = priority_key(&g);
        let seq = g.id;
        q.push(Queued { key, seq, item: g });
        drop(q);
        self.not_empty.notify_one();
        Ok(())
    }
    fn gather(&self, batch: usize) -> Option<Vec<Box<InFlight>>> {
        let mut q = self.inner.lock().unwrap();
        while q.len() < batch && !self.closed.load(AtomicOrdering::Relaxed) {
            q = self.not_empty.wait(q).unwrap();
        }
        if q.len() < batch {
            return None; // closed with a partial batch -> drop and shut down
        }
        let games: Vec<Box<InFlight>> = (0..batch).map(|_| q.pop().unwrap().item).collect();
        drop(q);
        self.not_full.notify_all();
        Some(games)
    }
    fn close(&self) {
        self.closed.store(true, AtomicOrdering::Relaxed);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>, // requeued games awaiting a worker
    queue_cv: Condvar,
    preinfer: PreInferQueue,
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    games_done: AtomicU64,
    rows_done: AtomicU64,
}

fn worker_loop(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) {
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        let mut g: Box<InFlight> = {
            let mut q = shared.queue.lock().unwrap();
            match q.pop() {
                Some(qd) => qd.item,
                None => {
                    drop(q);
                    Box::new(spawn_fresh(&shared))
                }
            }
        };
        if let Some(res) = g.pending.take() {
            apply_result(&mut g, res, &shared.config);
        }
        match advance_until_eval_or_done(&mut g, &shared.config) {
            Advance::Eval => {
                if shared.preinfer.push(g).is_err() {
                    return; // closed (shutdown)
                }
            }
            Advance::Done(rows) => {
                shared.games_done.fetch_add(1, AtomicOrdering::Relaxed);
                shared.rows_done.fetch_add(rows.len() as u64, AtomicOrdering::Relaxed);
                sink(rows);
            }
        }
    }
}

fn requeue_scattered(
    shared: &Shared,
    logits: Arc<Vec<f32>>,
    values: &[f32],
    games: Vec<Box<InFlight>>,
) {
    let mut q = shared.queue.lock().unwrap();
    for (r, mut g) in games.into_iter().enumerate() {
        g.pending = Some(EvalResult {
            logits: Arc::clone(&logits),
            row: r,
            value: values[r],
        });
        let key = priority_key(&g);
        let seq = g.id;
        q.push(Queued { key, seq, item: g });
    }
    drop(q);
    shared.queue_cv.notify_all();
}

// --- inference + scatter -----------------------------------------------------
struct WorkItem {
    logits: Tensor, // GPU [m, 33]
    values: Tensor, // GPU [m]
    event: CudaEvent,
    games: Vec<Box<InFlight>>,
}

/// Owns the tch net on the inference thread; reloads weights in place on mtime
/// change (the trainer republishes the safetensors sidecar atomically).
struct Infer {
    net: Net,
    dev: Device,
    weights_path: String,
    reload_every: Duration,
    last_check: Instant,
    last_mtime: Option<std::time::SystemTime>,
}
impl Infer {
    fn new(config: &Config, dev: Device) -> Infer {
        Infer {
            net: Net::load(&config.weights_path, dev, Kind::Float),
            dev,
            last_mtime: std::fs::metadata(&config.weights_path).and_then(|m| m.modified()).ok(),
            weights_path: config.weights_path.clone(),
            reload_every: config.reload_every,
            last_check: Instant::now(),
        }
    }
    fn maybe_reload(&mut self) {
        if self.last_check.elapsed() < self.reload_every {
            return;
        }
        self.last_check = Instant::now();
        let m = std::fs::metadata(&self.weights_path).and_then(|md| md.modified()).ok();
        if m.is_none() || m == self.last_mtime {
            return;
        }
        self.net.reload(&self.weights_path); // copies fresh weights in place
        self.last_mtime = m;
    }
    /// Launch one async forward over `m` staged encodings; returns the GPU output
    /// tensors + a completion event (the scatter thread reads them back).
    fn launch(&self, enc: &[f32], m: usize) -> (Tensor, Tensor, CudaEvent) {
        let x = Tensor::from_slice(enc)
            .reshape([m as i64, ENC_LEN as i64])
            .to_device(self.dev);
        let (logits, values) = self.net.forward(&x);
        let logits = logits.to_kind(Kind::Float);
        let values = values.to_kind(Kind::Float);
        let event = CudaEvent::new();
        event.record();
        (logits, values, event)
    }
}

fn inference_thread(shared: Arc<Shared>, work_tx: SyncSender<WorkItem>) {
    let dev = pick_device(shared.config.cpu);
    let mut infer = Infer::new(&shared.config, dev);
    let batch = shared.config.batch.max(1);
    let mut enc: Vec<f32> = Vec::with_capacity(batch * ENC_LEN);
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        let games = match shared.preinfer.gather(batch) {
            Some(g) => g,
            None => return, // closed: dropping work_tx drains the scatter thread
        };
        enc.clear();
        for g in &games {
            enc.extend_from_slice(&g.staged_enc);
        }
        infer.maybe_reload();
        let m = games.len();
        let (logits, values, event) = infer.launch(&enc, m);
        if work_tx.send(WorkItem { logits, values, event, games }).is_err() {
            return; // scatter gone
        }
    }
}

fn scatter_thread(shared: Arc<Shared>, work_rx: Receiver<WorkItem>) {
    while let Ok(item) = work_rx.recv() {
        item.event.sync();
        let m = item.games.len();
        let logits = item.logits.to_device(Device::Cpu).contiguous();
        let values = item.values.to_device(Device::Cpu).contiguous();
        let mut lh = vec![0f32; m * POLICY_SIZE];
        logits.copy_data(&mut lh, m * POLICY_SIZE);
        let mut vh = vec![0f32; m];
        values.copy_data(&mut vh, m);
        requeue_scattered(&shared, Arc::new(lh), &vh, item.games);
    }
}

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

fn make_shared(config: Config) -> Arc<Shared> {
    let cap = (2 * config.n_slots.max(1) * config.batch.max(1)).max(config.batch.max(1));
    Arc::new(Shared {
        queue: Mutex::new(BinaryHeap::new()),
        queue_cv: Condvar::new(),
        preinfer: PreInferQueue::new(cap),
        stop: AtomicBool::new(false),
        id_counter: AtomicU64::new(0),
        config,
        games_done: AtomicU64::new(0),
        rows_done: AtomicU64::new(0),
    })
}

struct Pipeline {
    workers: Vec<thread::JoinHandle<()>>,
    infer: thread::JoinHandle<()>,
    scatter: thread::JoinHandle<()>,
}

fn spawn_pipeline(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) -> Pipeline {
    let n_threads = shared.config.n_threads.max(1);
    let n_slots = shared.config.n_slots.max(1);
    let mut workers = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let sh = shared.clone();
        let sk = sink.clone();
        workers.push(thread::spawn(move || worker_loop(sh, sk)));
    }
    let (work_tx, work_rx) = sync_channel::<WorkItem>(n_slots);
    let infer = {
        let sh = shared.clone();
        thread::spawn(move || inference_thread(sh, work_tx))
    };
    let scatter = {
        let sh = shared.clone();
        thread::spawn(move || scatter_thread(sh, work_rx))
    };
    Pipeline { workers, infer, scatter }
}

fn shutdown(shared: &Shared, p: Pipeline) {
    shared.stop.store(true, AtomicOrdering::Relaxed);
    shared.preinfer.close();
    shared.queue_cv.notify_all();
    for h in p.workers {
        let _ = h.join();
    }
    let _ = p.infer.join();
    let _ = p.scatter.join();
}

// --- frame streaming (serve mode) --------------------------------------------
fn write_f32s<W: Write>(w: &mut W, s: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) };
    w.write_all(bytes)
}

/// One finished game = one frame: u32 n_rows, then n_rows x (state[230] f32,
/// pi[33] f32, z f32), little-endian. Flushed per game.
fn write_frame(out: &Mutex<BufWriter<io::Stdout>>, rows: &[Row]) {
    let mut w = out.lock().unwrap();
    let _ = w.write_all(&(rows.len() as u32).to_le_bytes());
    for (state, pi, z) in rows {
        let _ = write_f32s(&mut *w, state);
        let _ = write_f32s(&mut *w, pi);
        let _ = w.write_all(&z.to_le_bytes());
    }
    let _ = w.flush();
}

/// Serve mode: run forever, streaming finished games as framed bytes on stdout.
/// Stops when the parent closes our stdin (EOF = orchestrator died).
pub fn run_serve(config: Config) {
    let shared = make_shared(config);
    let stdout = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    let so = stdout.clone();
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(move |rows: Vec<Row>| write_frame(&so, &rows));
    let pipeline = spawn_pipeline(shared.clone(), sink);

    let mut buf = [0u8; 256];
    loop {
        match io::stdin().read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
    shutdown(&shared, pipeline);
    let _ = stdout.lock().unwrap().flush();
}

/// Bench mode: run for `run`, printing games/sec + rows/sec every `interval`.
///
/// Self-play starts as a synchronized cohort (all games begin near t=0 and finish
/// in waves), so per-interval rates oscillate wildly for the first minute. We
/// discard the first `warmup` and report a single cumulative average over the
/// remaining measure window — that's the number to compare across batch sizes.
pub fn run_bench(config: Config, run: Duration, interval: Duration, warmup: Duration) {
    let sims = config.sims;
    let (nt, ns, nb) = (config.n_threads, config.n_slots, config.batch);
    let shared = make_shared(config);
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(|_rows: Vec<Row>| {});
    let pipeline = spawn_pipeline(shared.clone(), sink);

    println!("selfplay_rs bench: batch={nb} threads={nt} slots={ns} sims={sims} warmup={:.0}s", warmup.as_secs_f64());
    let start = Instant::now();
    let mut win_start = start;
    let (mut last_g, mut last_r) = (0u64, 0u64);
    // Counters/clock at the end of warmup, for the cumulative steady-state average.
    let mut measure_start: Option<Instant> = None;
    let (mut base_g, mut base_r) = (0u64, 0u64);
    while start.elapsed() < run {
        thread::sleep(Duration::from_millis(200));
        if measure_start.is_none() && start.elapsed() >= warmup {
            measure_start = Some(Instant::now());
            base_g = shared.games_done.load(AtomicOrdering::Relaxed);
            base_r = shared.rows_done.load(AtomicOrdering::Relaxed);
        }
        if win_start.elapsed() >= interval {
            let dt = win_start.elapsed().as_secs_f64();
            let g = shared.games_done.load(AtomicOrdering::Relaxed);
            let r = shared.rows_done.load(AtomicOrdering::Relaxed);
            println!(
                "[+{:5.0}s] {:7.3} games/sec  ({:5.1} rows/game, {:6.0} rows/sec)",
                start.elapsed().as_secs_f64(),
                (g - last_g) as f64 / dt,
                (r - last_r) as f64 / (g - last_g).max(1) as f64,
                (r - last_r) as f64 / dt,
            );
            last_g = g;
            last_r = r;
            win_start = Instant::now();
        }
    }
    let g = shared.games_done.load(AtomicOrdering::Relaxed);
    let r = shared.rows_done.load(AtomicOrdering::Relaxed);
    shutdown(&shared, pipeline);
    match measure_start {
        Some(t0) => {
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "STEADY batch={nb}: {:7.3} games/sec  ({:5.1} rows/game, {:6.0} rows/sec) over {:.0}s after {:.0}s warmup",
                (g - base_g) as f64 / dt,
                (r - base_r) as f64 / (g - base_g).max(1) as f64,
                (r - base_r) as f64 / dt,
                dt,
                warmup.as_secs_f64(),
            );
        }
        None => println!("STEADY batch={nb}: run shorter than warmup; no measurement"),
    }
}
