//! Continuous search-free self-play pipeline (100% Rust, decision-batched).
//!
//! AWR data generation: NO search. Each decision is one batched GPU forward of
//! the true match state; the move is sampled straight from the policy head
//! (illegal slots masked out, logits annealed by the match temperature). Rows
//! carry (state, legal mask, action, z) for the Python AWR trainer
//! (advantage-weighted regression on the match outcome).
//!
//! Topology (inherited unchanged from the ISMCTS pipeline this replaces):
//!   - N worker threads each drive one game's next decision: apply the move
//!     sampled from the previous forward's logits, then stage the next
//!     decision's encoding, or stream the finished game's rows to the sink.
//!     Workers claim scattered games in small chunks from the ply-bucketed
//!     return queue, highest ply first — the fast lane that keeps
//!     nearly-finished games finishing and the completion stream smooth. An
//!     empty return queue => spawn a fresh game (dynamic game count,
//!     self-regulating to fill the pipe);
//!   - ONE bounded ply-priority pre-inference queue (most real-moves-made first),
//!     so games closest to finishing are evaluated soonest;
//!   - ONE inference thread: gathers BATCH decision encodings, runs the forward,
//!     records a CUDA event, hands the in-flight forward to the scatter thread,
//!     and loops to launch the next batch. It also mtime-polls the weights sidecar
//!     and reloads in place between batches (weight hot-reload);
//!   - ONE scatter thread: waits the event, reads the logits back to the host,
//!     attaches each game's result, and bulk-pushes the batch back into the
//!     return queue.
//!
//! Both shared queues are ply-bucketed (one Vec per ply count), NOT comparison
//! heaps: push is O(1) and bulk drains move whole buckets, so every lock hold is
//! tiny.
//!
//! vs ISMCTS self-play: a move costs 1 forward instead of ~sims, so games/sec
//! rises by ~two orders of magnitude and the consumer (trainer ingest), not the
//! GPU, becomes the side to keep fast.

use std::io::{self, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tch::{Device, Kind, Tensor};

use foxlite_core::encode::{encode_into, legal_mask, real_card_from_canon_index, INPUT_SIZE};
use foxlite_core::mcts::temperature;
use foxlite_core::{Phase, Player, State, NUM_CARDS};

use crate::aoti::AotiModel;
use crate::cuda_event::CudaEvent;
use crate::net::Net;

pub const ENC_LEN: usize = INPUT_SIZE; // 230
pub const POLICY_SIZE: usize = NUM_CARDS; // 33

/// One finished-game training row: (state[230], legal_mask[33], action, z).
pub type Row = (Vec<f32>, [f32; NUM_CARDS], u8, f32);

pub struct Config {
    pub seed: u64,
    pub n_threads: usize,
    pub n_slots: usize, // GPU forwards kept in flight (inference runs ahead of scatter)
    pub batch: usize,   // GPU forward width
    pub weights_path: String,
    pub model_path: String, // AOTInductor .pt2 (fused forward); empty -> eager tch forward
    pub reload_every: Duration,
    pub cpu: bool,
}

// --- per-game seeding (splitmix64; determinism not required, just spread) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game search-free self-play ------------------------------------------
struct EvalResult {
    logits: Arc<Vec<f32>>, // whole batch's policy [m * 33]; the worker slices its row
    row: usize,
}
/// One recorded decision; flushed to a row with z = ±1 when the match ends.
struct Decision {
    state_enc: Vec<f32>,
    mask: [f32; NUM_CARDS], // legal moves at the decision (canonical slots)
    action: u8,             // canonical index of the move taken
    seat: Player,
}
struct InFlight {
    state: State, // the TRUE match state
    mover: Player, // seat to move at the staged decision (= state.awaiting)
    rng: StdRng,
    decisions: Vec<Decision>,
    pending: Option<EvalResult>,
    staged_enc: Vec<f32>,          // encoding of the decision awaiting a forward
    staged_mask: [f32; NUM_CARDS], // legal mask of that decision
}

fn new_game(seed: u64) -> InFlight {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = State::new_match(&mut rng);
    let mover = state.awaiting.expect("fresh match awaits a leader");
    InFlight {
        state,
        mover,
        rng,
        decisions: Vec::new(),
        pending: None,
        staged_enc: Vec::new(),
        staged_mask: [0.0; NUM_CARDS],
    }
}

fn spawn_fresh(shared: &Shared) -> InFlight {
    let id = shared.id_counter.fetch_add(1, AtomicOrdering::Relaxed);
    new_game(mix64(shared.config.seed ^ mix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15))))
}

fn finalize_rows(g: &mut InFlight) -> Vec<Row> {
    let winner = g.state.match_winner().expect("finalize before MatchOver");
    g.decisions
        .drain(..)
        .map(|d| {
            let z = if d.seat == winner { 1.0f32 } else { -1.0f32 };
            (d.state_enc, d.mask, d.action, z)
        })
        .collect()
}

/// Sample a canonical card index from softmax(logits / temp) over the legal
/// slots. `temp <= 0` degenerates to argmax over the legal logits.
fn sample_masked<R: Rng + ?Sized>(
    logits: &[f32],
    mask: &[f32; NUM_CARDS],
    temp: f64,
    rng: &mut R,
) -> usize {
    let mut maxl = f64::NEG_INFINITY;
    let mut arg = usize::MAX;
    for i in 0..NUM_CARDS {
        if mask[i] > 0.0 {
            let l = logits[i] as f64;
            if l > maxl {
                maxl = l;
                arg = i;
            }
        }
    }
    assert!(arg != usize::MAX, "no legal move to sample");
    if temp <= 0.0 {
        return arg;
    }
    let mut w = [0.0f64; NUM_CARDS];
    let mut total = 0.0;
    for i in 0..NUM_CARDS {
        if mask[i] > 0.0 {
            let e = (((logits[i] as f64) - maxl) / temp).exp();
            w[i] = e;
            total += e;
        }
    }
    let r = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (i, &wi) in w.iter().enumerate() {
        if wi > 0.0 {
            acc += wi;
            if r < acc {
                return i;
            }
        }
    }
    arg
}

/// Apply the forward result for the staged decision: sample a move from the
/// legal-masked, temperature-annealed policy, record the decision, play it, and
/// advance the true state to the next decision (or to MatchOver). The value
/// head is unused here — z comes from the match outcome at finalize.
fn apply_result(g: &mut InFlight, res: EvalResult) {
    let logits = &res.logits[res.row * POLICY_SIZE..(res.row + 1) * POLICY_SIZE];
    // Selection temperature, annealed over the match (same schedule the ISMCTS
    // runs trained under); exploration comes solely from this sampling.
    let temp = temperature(g.state.round_num, g.state.trick_num);
    let action = sample_masked(logits, &g.staged_mask, temp, &mut g.rng);
    g.decisions.push(Decision {
        state_enc: std::mem::take(&mut g.staged_enc),
        mask: g.staged_mask,
        action: action as u8,
        seat: g.mover,
    });

    let card = real_card_from_canon_index(action, g.state.trump.suit);
    g.state.apply(card);
    loop {
        match g.state.phase {
            Phase::Playing | Phase::MatchOver => break,
            Phase::TrickComplete => g.state.advance_after_trick(),
            Phase::RoundOver => {
                let mut rng = std::mem::replace(&mut g.rng, StdRng::seed_from_u64(0));
                g.state.end_round(&mut rng);
                g.rng = rng;
            }
        }
    }
}

enum Advance {
    Eval,
    Done(Vec<Row>),
}

/// Stage the next decision for evaluation, or finalize a finished match.
fn advance_until_eval_or_done(g: &mut InFlight) -> Advance {
    if g.state.phase == Phase::MatchOver {
        return Advance::Done(finalize_rows(g));
    }
    g.mover = g.state.awaiting.expect("decision awaits a mover");
    encode_into(&g.state, g.mover, &mut g.staged_enc);
    g.staged_mask = legal_mask(&g.state, g.mover);
    Advance::Eval
}

// --- ply-bucketed priority queue (most real moves made first) -----------------
//
// The priority key is the number of real moves already made — a small bounded
// integer — so one Vec per ply replaces a comparison heap: push is O(1), and a
// batch gather drains whole buckets top-down (pointer memmoves, not per-item
// sift-downs), keeping every lock hold tiny.

/// A match is at most 7 rounds (every round awards >= 6 total points and 21
/// ends it) x 26 plies = 182 plies, and a game is re-queued having made at most
/// 181 moves — so ply keys span 0..=181. The min() clamp is belt-and-braces.
const N_BUCKETS: usize = 182;

fn priority_key(g: &InFlight) -> usize {
    g.decisions.len().min(N_BUCKETS - 1)
}

struct Buckets {
    by_ply: Vec<Vec<Box<InFlight>>>,
    len: usize,
    hi: usize, // highest possibly-non-empty bucket (raised on push, settled by drains)
}
impl Buckets {
    fn new() -> Buckets {
        Buckets { by_ply: (0..N_BUCKETS).map(|_| Vec::new()).collect(), len: 0, hi: 0 }
    }
    fn push(&mut self, g: Box<InFlight>) {
        let k = priority_key(&g);
        self.by_ply[k].push(g);
        self.len += 1;
        if k > self.hi {
            self.hi = k;
        }
    }
    /// Move up to `max` games from the highest buckets into `out` (newest first
    /// within a ply, mirroring the old heap's seq tie-break). Returns the count.
    fn drain_top(&mut self, max: usize, out: &mut Vec<Box<InFlight>>) -> usize {
        let want = max.min(self.len);
        let mut moved = 0;
        let mut b = self.hi;
        while moved < want {
            let take = (want - moved).min(self.by_ply[b].len());
            if take > 0 {
                let bucket = &mut self.by_ply[b];
                let at = bucket.len() - take;
                out.extend(bucket.drain(at..));
                self.len -= take;
                moved += take;
            }
            if moved == want || b == 0 {
                break;
            }
            b -= 1;
        }
        self.hi = b;
        moved
    }
}

struct PreInferState {
    buckets: Buckets,
    // Waiter bookkeeping so notifies are skipped (no futex syscall) when no one
    // is parked — the common case for both condvars.
    waiting_pushers: usize,
    gather_waiting: bool,
}

/// Bounded, blocking ply-priority pre-inference queue.
struct PreInferQueue {
    inner: Mutex<PreInferState>,
    batch_ready: Condvar, // inference thread waits here for len >= batch
    not_full: Condvar,    // workers wait here for len < cap
    cap: usize,
    batch: usize,
    closed: AtomicBool,
}
impl PreInferQueue {
    fn new(cap: usize, batch: usize) -> PreInferQueue {
        PreInferQueue {
            inner: Mutex::new(PreInferState {
                buckets: Buckets::new(),
                waiting_pushers: 0,
                gather_waiting: false,
            }),
            batch_ready: Condvar::new(),
            not_full: Condvar::new(),
            cap,
            batch,
            closed: AtomicBool::new(false),
        }
    }
    fn push(&self, g: Box<InFlight>) -> Result<(), ()> {
        let mut q = self.inner.lock().unwrap();
        while q.buckets.len >= self.cap && !self.closed.load(AtomicOrdering::Relaxed) {
            q.waiting_pushers += 1;
            q = self.not_full.wait(q).unwrap();
            q.waiting_pushers -= 1;
        }
        if self.closed.load(AtomicOrdering::Relaxed) {
            return Err(());
        }
        q.buckets.push(g);
        // Edge-triggered wake: gather re-checks `len` itself while it stays
        // >= batch, so signalling every push would be a wasted syscall per leaf.
        let wake = q.gather_waiting && q.buckets.len >= self.batch;
        drop(q);
        if wake {
            self.batch_ready.notify_one();
        }
        Ok(())
    }
    fn gather(&self) -> Option<Vec<Box<InFlight>>> {
        let batch = self.batch;
        let mut q = self.inner.lock().unwrap();
        while q.buckets.len < batch && !self.closed.load(AtomicOrdering::Relaxed) {
            q.gather_waiting = true;
            q = self.batch_ready.wait(q).unwrap();
            q.gather_waiting = false;
        }
        if q.buckets.len < batch {
            return None; // closed with a partial batch -> drop and shut down
        }
        let mut games: Vec<Box<InFlight>> = Vec::with_capacity(batch);
        q.buckets.drain_top(batch, &mut games);
        let wake = q.waiting_pushers > 0;
        drop(q);
        if wake {
            // At most n_threads waiters, each doing an O(1) push on wake; the
            // old per-push notify_one + post-gather broadcast was a herd.
            self.not_full.notify_all();
        }
        Some(games)
    }
    fn close(&self) {
        self.closed.store(true, AtomicOrdering::Relaxed);
        self.batch_ready.notify_all();
        self.not_full.notify_all();
    }
}

/// Games a worker claims from the return queue per lock hold. Large enough to
/// make the lock acquisition rate trivial (~2 per worker per batch cycle),
/// small enough that the ply fast-lane keeps cross-cycle granularity: a worker
/// re-checks the top buckets every CLAIM games, so nearly-finished games keep
/// jumping ahead and completions stay smooth. (v1 of this rework used plain
/// per-worker inboxes — no cross-game priority — and the whole in-flight
/// population marched in lockstep, finishing in synchronized waves that
/// transiently drained the pipe and cost ~6% games/sec.)
const CLAIM: usize = 32;

struct Shared {
    /// Scattered games (result attached) awaiting a worker, ply-bucketed:
    /// scatter bulk-pushes a batch in one short hold (O(1) per game), workers
    /// claim CLAIM-sized chunks from the top buckets.
    returns: Mutex<Buckets>,
    preinfer: PreInferQueue,
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    games_done: AtomicU64,
    rows_done: AtomicU64,
}

fn worker_loop(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) {
    let mut local: Vec<Box<InFlight>> = Vec::new(); // claimed chunk, highest ply last
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        let mut g: Box<InFlight> = match local.pop() {
            Some(g) => g,
            None => {
                {
                    let mut q = shared.returns.lock().unwrap();
                    q.drain_top(CLAIM, &mut local);
                }
                local.reverse(); // drain_top fills highest-first; pop() takes the back
                match local.pop() {
                    Some(g) => g,
                    // Nothing scattered anywhere => spawn rather than wait
                    // (dynamic game count; the preinfer cap is the backpressure).
                    None => Box::new(spawn_fresh(&shared)),
                }
            }
        };
        if let Some(res) = g.pending.take() {
            apply_result(&mut g, res);
        }
        match advance_until_eval_or_done(&mut g) {
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

fn requeue_scattered(shared: &Shared, logits: Arc<Vec<f32>>, mut games: Vec<Box<InFlight>>) {
    // Attach results outside the lock; the hold below is pure bucket pushes.
    for (r, g) in games.iter_mut().enumerate() {
        g.pending = Some(EvalResult { logits: Arc::clone(&logits), row: r });
    }
    let mut q = shared.returns.lock().unwrap();
    for g in games {
        q.push(g);
    }
}

// --- inference + scatter -----------------------------------------------------
struct WorkItem {
    logits: Tensor, // GPU [m, 33]
    // The value head is unused by search-free self-play; the tensor is carried
    // only to keep it alive until the event sync (same stream as the logits).
    values: Tensor, // GPU [m]
    event: CudaEvent,
    games: Vec<Box<InFlight>>,
}

/// The forward backend: the eager tch net (CPU / no `--model`), or the fused
/// AOTInductor `.pt2` (CUDA + `--model`). Both consume bf16 input on CUDA and
/// return bf16 outputs; the launch path narrows/widens identically for either.
enum Backend {
    Tch(Net),
    Aoti(AotiModel),
}

/// Push the safetensors sidecar's weights into the loaded AOTI package in place
/// (no recompile). Constants are keyed by the MANGLED FQN ('.'->'_'); tensors go
/// to CUDA bf16. Returns false on a transient read error (retried next tick).
fn aoti_swap(model: &AotiModel, weights_path: &str, dev: Device) -> bool {
    let entries = match Tensor::read_safetensors(weights_path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut names: Vec<std::ffi::CString> = Vec::with_capacity(entries.len());
    let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
    for (name, t) in entries {
        names.push(std::ffi::CString::new(name.replace('.', "_")).unwrap());
        tensors.push(t.to_device(dev).to_kind(Kind::BFloat16));
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    model.swap_weights(&names, &refs);
    true
}

/// Owns the forward backend on the inference thread; reloads weights in place on
/// mtime change (the trainer republishes the safetensors sidecar atomically).
struct Infer {
    backend: Backend,
    dev: Device,
    kind: Kind, // forward dtype: bf16 on CUDA (tensor cores + half the memory traffic), fp32 on CPU
    weights_path: String,
    reload_every: Duration,
    last_check: Instant,
    last_mtime: Option<std::time::SystemTime>,
}
impl Infer {
    fn new(config: &Config, dev: Device) -> Infer {
        // The forward is dominated by matmul; bf16 uses tensor cores + halves the
        // H2D/activation traffic. CPU bf16 is slow/uneven, so CPU stays fp32.
        let kind = if matches!(dev, Device::Cuda(_)) { Kind::BFloat16 } else { Kind::Float };
        let use_aoti = !config.model_path.is_empty() && matches!(dev, Device::Cuda(_));
        let backend = if use_aoti {
            // Loaded ONCE; the .pt2 bakes export-time weights, so swap in the current
            // sidecar before the first forward (and on every publish via maybe_reload).
            let model = AotiModel::load(&config.model_path);
            match model.check_batch(config.batch as i64, ENC_LEN as i64) {
                0 => {}
                code => {
                    eprintln!(
                        "FATAL: {} not compiled for batch={} (aoti_check_batch={code}); \
                         re-export the .pt2 at this batch (SERVING_BATCH in net.py)",
                        config.model_path, config.batch
                    );
                    std::process::exit(1);
                }
            }
            aoti_swap(&model, &config.weights_path, dev);
            Backend::Aoti(model)
        } else {
            Backend::Tch(Net::load(&config.weights_path, dev, kind))
        };
        Infer {
            backend,
            dev,
            kind,
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
        match &self.backend {
            Backend::Tch(net) => net.reload(&self.weights_path), // copies fresh weights in place
            Backend::Aoti(model) => {
                if !aoti_swap(model, &self.weights_path, self.dev) {
                    return; // transient read; keep last_mtime so we retry next tick
                }
            }
        }
        self.last_mtime = m;
    }
    /// Launch one async forward over `m` staged encodings; returns the GPU output
    /// tensors + a completion event (the scatter thread reads them back).
    fn launch(&self, enc: &[f32], m: usize) -> (Tensor, Tensor, CudaEvent) {
        // Narrow to the forward dtype on the CPU side so the H2D copy moves half
        // the bytes (bf16), then ship to the device.
        let x = Tensor::from_slice(enc)
            .reshape([m as i64, ENC_LEN as i64])
            .to_kind(self.kind)
            .to_device(self.dev);
        let (logits, values) = match &self.backend {
            Backend::Tch(net) => net.forward(&x),
            Backend::Aoti(model) => {
                // Fresh output tensors per launch (not static buffers), so the scatter
                // thread can read these while the next forward runs — same overlap as
                // the eager path. AOTI is static-batch, so m must equal config.batch
                // (guaranteed: gather() only returns full batches).
                let out_l = Tensor::zeros([m as i64, POLICY_SIZE as i64], (self.kind, self.dev));
                let out_v = Tensor::zeros([m as i64], (self.kind, self.dev));
                model.run(&x, &out_l, &out_v);
                (out_l, out_v)
            }
        };
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
        let games = match shared.preinfer.gather() {
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
        let mut lh = vec![0f32; m * POLICY_SIZE];
        logits.copy_data(&mut lh, m * POLICY_SIZE);
        requeue_scattered(&shared, Arc::new(lh), item.games);
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
    let batch = config.batch.max(1);
    let cap = (2 * config.n_slots.max(1) * batch).max(batch);
    Arc::new(Shared {
        returns: Mutex::new(Buckets::new()),
        preinfer: PreInferQueue::new(cap, batch),
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
/// legal_mask[33] f32, action f32, z f32), little-endian. Flushed per game.
fn write_frame(out: &Mutex<BufWriter<io::Stdout>>, rows: &[Row]) {
    let mut w = out.lock().unwrap();
    let _ = w.write_all(&(rows.len() as u32).to_le_bytes());
    for (state, mask, action, z) in rows {
        let _ = write_f32s(&mut *w, state);
        let _ = write_f32s(&mut *w, mask);
        let _ = w.write_all(&(*action as f32).to_le_bytes());
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
    let (nt, ns, nb) = (config.n_threads, config.n_slots, config.batch);
    let shared = make_shared(config);
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(|_rows: Vec<Row>| {});
    let pipeline = spawn_pipeline(shared.clone(), sink);

    println!("selfplay_rs bench: batch={nb} threads={nt} slots={ns} warmup={:.0}s", warmup.as_secs_f64());
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive full games through the worker-side primitives with a dummy uniform
    /// net (no GPU): every recorded action must be legal under its row's mask,
    /// the mask must match the true state's legal moves at staging time, and z
    /// must be ±1 partitioned by the winning seat.
    #[test]
    fn selfplay_rows_are_legal_and_consistent() {
        for seed in 0..40u64 {
            let mut g = new_game(seed);
            let rows = loop {
                match advance_until_eval_or_done(&mut g) {
                    Advance::Eval => {
                        assert_eq!(g.staged_enc.len(), ENC_LEN);
                        assert_eq!(g.staged_mask, legal_mask(&g.state, g.mover));
                        let res = EvalResult {
                            logits: Arc::new(vec![0.0f32; POLICY_SIZE]),
                            row: 0,
                        };
                        apply_result(&mut g, res);
                    }
                    Advance::Done(rows) => break rows,
                }
            };
            assert!(!rows.is_empty(), "match produced no decisions (seed {seed})");
            for (state, mask, action, z) in &rows {
                assert_eq!(state.len(), ENC_LEN);
                assert_eq!(mask[*action as usize], 1.0, "illegal action sampled (seed {seed})");
                assert!(*z == 1.0 || *z == -1.0);
            }
            assert!(rows.iter().any(|r| r.3 == 1.0) && rows.iter().any(|r| r.3 == -1.0),
                    "both seats decide in every match (seed {seed})");
        }
    }

    #[test]
    fn sample_masked_respects_mask_and_argmax_at_temp_zero() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut mask = [0.0f32; NUM_CARDS];
        mask[3] = 1.0;
        mask[10] = 1.0;
        mask[20] = 1.0;
        let mut logits = [0.0f32; NUM_CARDS];
        logits[5] = 100.0; // illegal: huge logit must never be picked
        logits[10] = 2.0;
        let mut seen = [0u32; NUM_CARDS];
        for _ in 0..300 {
            let a = sample_masked(&logits, &mask, 1.0, &mut rng);
            assert!(mask[a] > 0.0, "sampled an illegal slot");
            seen[a] += 1;
        }
        assert!(seen[10] > seen[3] && seen[10] > seen[20], "higher logit should dominate");
        assert!(seen[3] > 0 && seen[20] > 0, "temp 1.0 should still explore");
        assert_eq!(sample_masked(&logits, &mask, 0.0, &mut rng), 10);
    }
}
