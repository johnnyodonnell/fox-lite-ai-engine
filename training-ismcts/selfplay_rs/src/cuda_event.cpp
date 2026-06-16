// Minimal C FFI over libtorch's CUDA event API (not exposed by tch-rs). The
// inference thread records an event on the current CUDA stream right after it
// launches a forward; the scatter thread syncs that event before reading the
// outputs back, so the D2H readback overlaps the inference thread's next forward.
//
// Symbols resolve against the c10_cuda / cudart libs that build.rs already links
// (--no-as-needed -lc10_cuda -ltorch_cuda -l:libcudart.so.12).
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" {

void* cuda_event_new() {
    cudaEvent_t e;
    // BlockingSync: cudaEventSynchronize parks the waiting (scatter) thread on an
    // OS wait — woken by the completion interrupt — instead of spin-burning a core.
    // DisableTiming: we only need ordering, not elapsed-time queries.
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync);
    return reinterpret_cast<void*>(e);
}
void cuda_event_record(void* e) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(e),
                    c10::cuda::getCurrentCUDAStream().stream());
}
void cuda_event_sync(void* e) {
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(e));
}
void cuda_event_free(void* e) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e));
}

}  // extern "C"
