// Link the box's PyTorch libtorch (via LIBTORCH_USE_PYTORCH=1) and force-load the
// CUDA backend libs so tch::Cuda::is_available() is true. Without --no-as-needed
// the linker drops torch_cuda from binaries that reference no CUDA symbol directly
// (the evaluator), its static initializers never run, and CUDA looks unavailable.
//
// Also compile a tiny CUDA-event FFI shim (src/cuda_event.cpp) so the self-play
// pipeline can overlap each forward's readback with the next launch (scatter
// thread). It only needs cudaEvent + c10::cuda::getCurrentCUDAStream — no
// CUDA-graph / AOTI machinery (unlike chess-ai-engine).
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
    println!("cargo:rerun-if-changed=src/cuda_event.cpp");

    // --- compile the CUDA-event shim against the box's libtorch + CUDA headers ---
    let includes =
        py_out("import torch.utils.cpp_extension as c; print('\\n'.join(c.include_paths()))");
    // torch headers #include "crt/host_config.h", absent from the pip
    // nvidia-cuda-runtime wheel — pick a CUDA include dir that has both
    // cuda_runtime.h and crt/ (the triton-bundled cu12 set qualifies).
    let cuda_inc = py_out(
        "import os, sysconfig, glob\n\
         sp = sysconfig.get_paths()['purelib']\n\
         hits = glob.glob(os.path.join(sp, '**/include/crt/host_config.h'), recursive=True)\n\
         print(os.path.dirname(os.path.dirname(hits[0])) if hits else '/usr/local/cuda/include')",
    );
    let cxx11 = py_out("import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)");

    let mut b = cc::Build::new();
    b.cpp(true).std("c++17").file("src/cuda_event.cpp");
    for line in includes.lines() {
        let p = line.trim();
        if !p.is_empty() {
            b.include(p);
        }
    }
    b.include(cuda_inc.trim());
    b.flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11.trim()));
    b.flag("-DTORCH_API_INCLUDE_EXTENSION_H");
    b.flag_if_supported("-Wno-unused-parameter");
    b.compile("cuda_event_shim");

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
