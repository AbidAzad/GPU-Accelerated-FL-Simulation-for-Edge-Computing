// fedavg_gpu_windows.cu
// Windows DLL version of FedAvg GPU aggregator with persistent allocations.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------
// Simple CUDA error check helper
// ---------------------------------------------------------------------
static inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[fedavg_gpu] %s failed: %s\n",
                     msg, cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------
// Persistent device buffers
// ---------------------------------------------------------------------
static float* d_clients = nullptr;   // [num_clients * vec_len]
static float* d_out     = nullptr;   // [vec_len]
static int*   d_counts  = nullptr;   // [num_clients]

static int    allocated_clients = 0;
static int    allocated_vec_len = 0;


// ---------------------------------------------------------------------
// Kernel: weighted average over clients for each parameter index
// out[j] = sum_c counts[c] * clients[c * vec_len + j] / total_count
// ---------------------------------------------------------------------
__global__ void fedavg_kernel(const float* __restrict__ clients,
                              const int*   __restrict__ counts,
                              float*       __restrict__ out,
                              int num_clients,
                              int vec_len,
                              int total_count) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= vec_len) return;

    float acc = 0.0f;
    for (int c = 0; c < num_clients; ++c) {
        float w = clients[c * vec_len + j];
        acc += static_cast<float>(counts[c]) * w;
    }
    out[j] = acc / static_cast<float>(total_count);
}


// ---------------------------------------------------------------------
// Ensure we have enough device memory for given (num_clients, vec_len).
// Reallocates only when we need more capacity.
// ---------------------------------------------------------------------
static void ensure_capacity(int num_clients, int vec_len) {
    if (num_clients <= 0 || vec_len <= 0) {
        return;
    }

    bool need_realloc = false;

    if (!d_clients || !d_out || !d_counts) {
        need_realloc = true;
    } else if (num_clients > allocated_clients || vec_len > allocated_vec_len) {
        need_realloc = true;
    }

    if (!need_realloc) return;

    // Free old buffers if they exist
    if (d_clients) { check_cuda(cudaFree(d_clients), "cudaFree(d_clients)"); d_clients = nullptr; }
    if (d_counts)  { check_cuda(cudaFree(d_counts),  "cudaFree(d_counts)");  d_counts  = nullptr; }
    if (d_out)     { check_cuda(cudaFree(d_out),     "cudaFree(d_out)");     d_out     = nullptr; }

    // Allocate new buffers sized for the current maximum
    size_t clients_bytes = static_cast<size_t>(num_clients) *
                           static_cast<size_t>(vec_len) *
                           sizeof(float);
    size_t counts_bytes  = static_cast<size_t>(num_clients) * sizeof(int);
    size_t out_bytes     = static_cast<size_t>(vec_len) * sizeof(float);

    check_cuda(cudaMalloc(&d_clients, clients_bytes), "cudaMalloc(d_clients)");
    check_cuda(cudaMalloc(&d_counts,  counts_bytes),  "cudaMalloc(d_counts)");
    check_cuda(cudaMalloc(&d_out,     out_bytes),     "cudaMalloc(d_out)");

    allocated_clients = num_clients;
    allocated_vec_len = vec_len;
}


// ---------------------------------------------------------------------
// Exported function called from Python (via ctypes).
// Signature must match fl_core._load_fedavg_lib expectations:
//   extern "C" void fedavg_weighted_average_gpu(
//       const float* h_client_weights,
//       const int*   h_counts,
//       float*       h_out,
//       int          num_clients,
//       int          vec_len);
// ---------------------------------------------------------------------
extern "C" __declspec(dllexport)
void fedavg_weighted_average_gpu(const float* h_client_weights,
                                 const int*   h_counts,
                                 float*       h_out,
                                 int          num_clients,
                                 int          vec_len) {
    if (!h_client_weights || !h_counts || !h_out ||
        num_clients <= 0 || vec_len <= 0) {
        std::fprintf(stderr,
                     "[fedavg_gpu] Invalid arguments: clients=%p, counts=%p, "
                     "out=%p, num_clients=%d, vec_len=%d\n",
                     (void*)h_client_weights,
                     (void*)h_counts,
                     (void*)h_out,
                     num_clients,
                     vec_len);
        return;
    }

    // Lazily allocate / grow persistent device buffers
    ensure_capacity(num_clients, vec_len);

    // Copy host data → device
    size_t clients_bytes = static_cast<size_t>(num_clients) *
                           static_cast<size_t>(vec_len) *
                           sizeof(float);
    size_t counts_bytes  = static_cast<size_t>(num_clients) * sizeof(int);

    check_cuda(cudaMemcpy(d_clients, h_client_weights,
                          clients_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H→D (clients)");

    check_cuda(cudaMemcpy(d_counts, h_counts,
                          counts_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H→D (counts)");

    // Compute total_count on host
    long long total_count_ll = 0;
    for (int i = 0; i < num_clients; ++i) {
        total_count_ll += static_cast<long long>(h_counts[i]);
    }
    if (total_count_ll <= 0) {
        std::fprintf(stderr,
                     "[fedavg_gpu] total_count <= 0, skipping aggregation.\n");
        return;
    }
    int total_count = static_cast<int>(total_count_ll);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (vec_len + threadsPerBlock - 1) / threadsPerBlock;
    fedavg_kernel<<<blocks, threadsPerBlock>>>(
        d_clients, d_counts, d_out,
        num_clients, vec_len, total_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[fedavg_gpu] Kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return;
    }

    // Make sure kernel finished before copying results
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    // Copy result back to host
    size_t out_bytes = static_cast<size_t>(vec_len) * sizeof(float);
    check_cuda(cudaMemcpy(h_out, d_out,
                          out_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D→H (out)");
}


// ---------------------------------------------------------------------
// Optional explicit cleanup.
// Call this once at the very end of training if you want to free GPU
// memory before process exit (otherwise CUDA runtime will clean up).
// ---------------------------------------------------------------------
extern "C" __declspec(dllexport)
void fedavg_gpu_release() {
    if (d_clients) { check_cuda(cudaFree(d_clients), "cudaFree(d_clients)"); d_clients = nullptr; }
    if (d_counts)  { check_cuda(cudaFree(d_counts),  "cudaFree(d_counts)");  d_counts  = nullptr; }
    if (d_out)     { check_cuda(cudaFree(d_out),     "cudaFree(d_out)");     d_out     = nullptr; }

    allocated_clients = 0;
    allocated_vec_len = 0;

    std::fprintf(stderr, "[fedavg_gpu] device buffers released.\n");
}
