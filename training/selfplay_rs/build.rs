// Link the box's PyTorch libtorch (via LIBTORCH_USE_PYTORCH=1) and force-load the
// CUDA backend libs so tch::Cuda::is_available() is true. Without --no-as-needed
// the linker drops torch_cuda from binaries that reference no CUDA symbol directly
// (the evaluator), its static initializers never run, and CUDA looks unavailable.
//
// No CUDA-graph shim here (unlike chess-ai-engine) — this MLP needs none.
use std::process::Command;

fn py() -> String {
    std::env::var("SELFPLAY_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn py_out(code: &str) -> String {
    let o = Command::new(py())
        .args(["-c", code])
        .output()
        .expect("failed to run python for build config (set SELFPLAY_PYTHON to the venv python)");
    if !o.status.success() {
        panic!("python build query failed: {}", String::from_utf8_lossy(&o.stderr));
    }
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let torch_lib =
        py_out("import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))");
    let cudart_dir = py_out(
        "import os, nvidia.cuda_runtime as r; print(os.path.join(os.path.dirname(r.__file__), 'lib'))",
    );
    println!("cargo:rustc-link-search=native={}", torch_lib.trim());
    println!("cargo:rustc-link-search=native={}", cudart_dir.trim());
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-lc10_cuda");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-l:libcudart.so.12");
}
