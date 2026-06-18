//! Rust wrapper over the AOTInductor package-loader FFI shim (`aoti.cpp`).
//!
//! Loads a `.pt2` (Inductor's fused Triton cubins, no Python) and runs the fused
//! forward. tch tensors bridge as raw `C_tensor` handles (`at::Tensor*` under the
//! hood); outputs are copied into caller-owned tch tensors. Weights hot-swap in
//! place (no recompile) via `swap_weights`.

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_long};

use tch::Tensor;

extern "C" {
    fn aoti_load(path: *const c_char) -> *mut c_void;
    fn aoti_run(loader: *mut c_void, input: *const c_void, out_logits: *const c_void, out_values: *const c_void, out_belief: *const c_void);
    fn aoti_check_batch(loader: *mut c_void, batch: c_long, input_size: c_long) -> i32;
    fn aoti_free(loader: *mut c_void);
    fn aoti_swap_weights(loader: *mut c_void, names: *const *const c_char, tensors: *const *const c_void, n: i32);
}

pub struct AotiModel {
    raw: *mut c_void,
}

// Used only from the single inference thread.
unsafe impl Send for AotiModel {}

impl AotiModel {
    pub fn load(path: &str) -> AotiModel {
        let c = CString::new(path).unwrap();
        let raw = unsafe { aoti_load(c.as_ptr()) };
        assert!(!raw.is_null(), "aoti_load returned null for {path}");
        AotiModel { raw }
    }

    /// Run the fused forward on `input` [B, INPUT_SIZE] bf16 CUDA; write results
    /// into `out_logits` [B,33], `out_values` [B], and `out_belief` [B,33]
    /// (caller-owned, bf16 CUDA).
    pub fn run(&self, input: &Tensor, out_logits: &Tensor, out_values: &Tensor, out_belief: &Tensor) {
        unsafe {
            aoti_run(
                self.raw,
                input.as_ptr() as *const c_void,
                out_logits.as_ptr() as *const c_void,
                out_values.as_ptr() as *const c_void,
                out_belief.as_ptr() as *const c_void,
            );
        }
    }

    /// Startup guard / warmup: probe whether the package accepts `[batch, input_size]`.
    /// Returns 0 = ok, 1 = forward threw (batch mismatch / other), 2 = output batch
    /// mismatch (see `aoti_check_batch` in aoti.cpp).
    pub fn check_batch(&self, batch: i64, input_size: i64) -> i32 {
        unsafe { aoti_check_batch(self.raw, batch as c_long, input_size as c_long) }
    }

    /// Refresh weights in place (no recompile): push the given constants into the
    /// inactive buffer, re-derive folded constants, swap live. `names` are MANGLED
    /// constant names (dotted FQN with '.'->'_'); `tensors` the matching CUDA bf16
    /// tensors. Must be called from the inference thread (single reader, between
    /// forwards).
    pub fn swap_weights(&self, names: &[CString], tensors: &[&Tensor]) {
        assert_eq!(names.len(), tensors.len(), "names/tensors length mismatch");
        let name_ptrs: Vec<*const c_char> = names.iter().map(|c| c.as_ptr()).collect();
        let tensor_ptrs: Vec<*const c_void> =
            tensors.iter().map(|t| t.as_ptr() as *const c_void).collect();
        unsafe {
            aoti_swap_weights(self.raw, name_ptrs.as_ptr(), tensor_ptrs.as_ptr(), names.len() as i32);
        }
    }
}

impl Drop for AotiModel {
    fn drop(&mut self) {
        unsafe { aoti_free(self.raw) }
    }
}
