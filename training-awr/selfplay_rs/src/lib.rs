//! selfplay_rs — Rust self-play worker + evaluator for fox-lite-ai-engine.

// The self-play workers allocate from many threads (per-decision encode /
// rows); glibc malloc arena contention was ~15% of worker on-CPU time back in
// the ISMCTS days, and mimalloc stays a cheap win.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod aoti;
pub mod cuda_event;
pub mod net;
pub mod pipeline;
