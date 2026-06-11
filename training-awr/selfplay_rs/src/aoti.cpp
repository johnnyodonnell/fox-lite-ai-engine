// C FFI over libtorch's AOTInductor package loader (the fused, Python-free
// forward), not exposed by tch-rs. Symbols resolve against the torch libs that
// tch links (LIBTORCH_USE_PYTORCH=1) + the c10_cuda/torch_cuda/cudart that
// build.rs force-loads. Mirrors chess-ai-engine's shim, adapted to FoxNet's
// [B, INPUT_SIZE] input and (policy[B,33], value[B]) outputs.
#include <ATen/Context.h>
#include <ATen/Functions.h> // at::zeros for the batch-guard probe
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {

// tch tensor handles (C_tensor*) ARE at::Tensor* under the hood, so we
// reinterpret them directly. Outputs are copy_'d into Rust-owned tensors so all
// lifetime stays on the tch side.
void* aoti_load(const char* path) {
    // run_single_threaded=true runs inline on the calling thread (no AOTI thread
    // pool). Has no downside for our single-stream inference and is capture-safe.
    return reinterpret_cast<void*>(new torch::inductor::AOTIModelPackageLoader(
        std::string(path), "model", /*run_single_threaded=*/true));
}

void aoti_run(void* loader_, const void* in_, void* out_logits_, void* out_values_) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    const at::Tensor& in = *reinterpret_cast<const at::Tensor*>(in_);
    std::vector<at::Tensor> outs = loader->run({in});
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_logits_))->copy_(outs[0]);
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_values_))->copy_(outs[1]);
}

void aoti_free(void* loader_) {
    delete reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
}

// Startup guard: verify the package was compiled for `batch` (it is exported with
// a STATIC batch, so a wrong batch trips the AOTI input-shape check, which throws).
// We catch here because throwing across extern "C" into Rust is UB. 0 = ok,
// 1 = forward threw, 2 = output batch mismatch. Doubles as a warmup forward.
int aoti_check_batch(void* loader_, long batch, long input_size) {
    try {
        auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
        auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
        at::Tensor x = at::zeros({static_cast<int64_t>(batch), static_cast<int64_t>(input_size)}, opts);
        std::vector<at::Tensor> outs = loader->run({x});
        return (!outs.empty() && outs[0].size(0) == batch) ? 0 : 2;
    } catch (...) {
        return 1;
    }
}

// In-place weight refresh (no recompile). names[] are MANGLED constant names
// (dotted FQN with '.'->'_'); tensors[] are tch C_tensor handles (at::Tensor*) on
// CUDA bf16. Push into the INACTIVE buffer, re-derive folded constants (needs the
// .pt2 compiled with use_runtime_constant_folding=True), then swap live. Runs on
// the caller's current stream, so it serializes against the inference thread's
// forwards (single reader; the active buffer is untouched until swap).
void aoti_swap_weights(void* loader_, const char** names, const void** tensors, int n) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    auto* runner = loader->get_runner();
    std::unordered_map<std::string, at::Tensor> map;
    map.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        map[std::string(names[i])] = *reinterpret_cast<const at::Tensor*>(tensors[i]);
    }
    runner->update_constant_buffer(map, /*use_inactive=*/true, /*validate_full_updates=*/false);
    auto stream = reinterpret_cast<AOTInductorStreamHandle>(
        c10::cuda::getCurrentCUDAStream().stream());
    runner->run_const_fold(/*use_inactive=*/true, stream);
    runner->swap_constant_buffer();
}

}  // extern "C"
