//! selfplay_rs — Rust self-play worker + evaluator for fox-lite-ai-engine.

// The MCTS workers allocate heavily from many threads (per-sim determinize /
// walk / encode); glibc malloc arena contention was ~15% of worker on-CPU time.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod aoti;
pub mod cuda_event;
pub mod net;
pub mod pipeline;
