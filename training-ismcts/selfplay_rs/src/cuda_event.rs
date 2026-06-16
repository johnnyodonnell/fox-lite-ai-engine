//! Thin wrapper over the CUDA-event FFI shim (`cuda_event.cpp`).
//!
//! The single inference thread records an event on the current CUDA stream after
//! launching a forward, then hands the event (with the GPU output tensor) to the
//! scatter thread. The scatter thread `sync`s the event before its D2H readback,
//! so the readback of forward N overlaps the inference thread's launch of N+1 —
//! the GPU stays fed instead of going idle between forwards.

use std::ffi::c_void;

extern "C" {
    fn cuda_event_new() -> *mut c_void;
    fn cuda_event_record(e: *mut c_void);
    fn cuda_event_sync(e: *mut c_void);
    fn cuda_event_free(e: *mut c_void);
}

/// A CUDA event recorded on the current stream; another thread can `sync` it to
/// wait for that stream's work (the forward) to finish.
pub struct CudaEvent {
    raw: *mut c_void,
}

// Recorded on the inference thread, waited on the scatter thread.
unsafe impl Send for CudaEvent {}

impl CudaEvent {
    pub fn new() -> CudaEvent {
        CudaEvent { raw: unsafe { cuda_event_new() } }
    }
    pub fn record(&self) {
        unsafe { cuda_event_record(self.raw) }
    }
    pub fn sync(&self) {
        unsafe { cuda_event_sync(self.raw) }
    }
}

impl Default for CudaEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cuda_event_free(self.raw) }
    }
}
