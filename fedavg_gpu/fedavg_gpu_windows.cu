// fedavg_gpu_windows.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// ---------------------------------------------------------------------
// Simple CUDA error check helper
// ---------------------------------------------------------------------
static inline void check_cuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[fedavg_gpu] %s failed: %s\n",
                     msg, cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------
// Persistent device buffers + capacity bookkeeping
// ---------------------------------------------------------------------
static float* d_clients = nullptr;   // (allocated_clients * allocated_vec_len)
static int*   d_counts  = nullptr;   // (allocated_clients)
static float* d_out     = nullptr;   // (allocated_vec_len)

static int    allocated_clients = 0;
static int    allocated_vec_len = 0;

// Ensure we have enough device memory for a given (num_clients, vec_len).
// Only reallocates when capacity needs to grow.
// ---------------------------------------------------------------------
static void ensure_capacity(int num_clients, int vec_len)
{
    if (num_clients <= 0 || vec_len <= 0) {
        return;
    }

    bool need_realloc = false;

    if (!d_clients || !d_counts || !d_out) {
        need_realloc = true;
    } else if (num_clients > allocated_clients || vec_len > allocated_vec_len) {
        need_realloc = true;
    }

    if (!need_realloc) {
        return;  // existing buffers are large enough
    }

    // Free any existing buffers
    if (d_clients) { check_cuda(cudaFree(d_clients), "cudaFree(d_clients)"); d_clients = nullptr; }
    if (d_counts)  { check_cuda(cudaFree(d_counts),  "cudaFree(d_counts)");  d_counts  = nullptr; }
    if (d_out)     { check_cuda(cudaFree(d_out),     "cudaFree(d_out)");     d_out     = nullptr; }

    // Allocate with the new capacity
    std::size_t clients_bytes = static_cast<std::size_t>(num_clients) *
                                static_cast<std::size_t>(vec_len) *
                                sizeof(float);
    std::size_t counts_bytes  = static_cast<std::size_t>(num_clients) *
                                sizeof(int);
    std::size_t out_bytes     = static_cast<std::size_t>(vec_len) *
                                sizeof(float);

    check_cuda(cudaMalloc(&d_clients, clients_bytes), "cudaMalloc(d_clients)");
    check_cuda(cudaMalloc(&d_counts,  counts_bytes),  "cudaMalloc(d_counts)");
    check_cuda(cudaMalloc(&d_out,     out_bytes),     "cudaMalloc(d_out)");

    allocated_clients = num_clients;
    allocated_vec_len = vec_len;

    std::fprintf(stderr,
                 "[fedavg_gpu] (re)allocated device buffers: "
                 "clients=%d, vec_len=%d (bytes: clients=%zu, counts=%zu, out=%zu)\n",
                 allocated_clients, allocated_vec_len,
                 clients_bytes, counts_bytes, out_bytes);
}

// ---------------------------------------------------------------------
// CUDA kernel: each thread computes one parameter index j in [0, vec_len).
// For that parameter, it loops over all clients and accumulates:
//
//    sum_i client_weights[i, j] * counts[i]
//
// Then divides by total_count to obtain the weighted average.
// ---------------------------------------------------------------------
__global__ void fedavg_kernel(
    const float* __restrict__ d_clients,  // (num_clients * vec_len)
    const int*   __restrict__ d_counts,   // (num_clients)
    float*       __restrict__ d_out,      // (vec_len)
    int num_clients,
    int vec_len,
    int total_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= vec_len) {
        return;
    }

    float acc = 0.0f;

    // clients are laid out row-major: [client][param]
    // so index = i * vec_len + j
    for (int i = 0; i < num_clients; ++i) {
        int idx = i * vec_len + j;
        acc += d_clients[idx] * static_cast<float>(d_counts[i]);
    }

    d_out[j] = acc / static_cast<float>(total_count);
}

// ---------------------------------------------------------------------
// Exported function called from Python (via ctypes).
// ---------------------------------------------------------------------
extern "C" __declspec(dllexport)
void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len)
{
    if (!h_client_weights || !h_counts || !h_out) {
        std::fprintf(stderr,
                     "[fedavg_gpu] received null host pointer(s); "
                     "aborting aggregation.\n");
        return;
    }

    if (num_clients <= 0 || vec_len <= 0) {
        std::fprintf(stderr,
                     "[fedavg_gpu] num_clients <= 0 or vec_len <= 0; "
                     "aborting aggregation.\n");
        return;
    }

    // Make sure device buffers are ready
    ensure_capacity(num_clients, vec_len);

    // Compute total_count on host with 64-bit accumulator to avoid overflow
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

    std::size_t clients_bytes = static_cast<std::size_t>(num_clients) *
                                static_cast<std::size_t>(vec_len) *
                                sizeof(float);
    std::size_t counts_bytes  = static_cast<std::size_t>(num_clients) *
                                sizeof(int);
    std::size_t out_bytes     = static_cast<std::size_t>(vec_len) *
                                sizeof(float);

    // Copy host data to device
    check_cuda(cudaMemcpy(d_clients,
                          h_client_weights,
                          clients_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D (clients)");

    check_cuda(cudaMemcpy(d_counts,
                          h_counts,
                          counts_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D (counts)");

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (vec_len + threadsPerBlock - 1) / threadsPerBlock;

    fedavg_kernel<<<blocks, threadsPerBlock>>>(
        d_clients,
        d_counts,
        d_out,
        num_clients,
        vec_len,
        total_count
    );

    check_cuda(cudaGetLastError(), "fedavg_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result back to host
    check_cuda(cudaMemcpy(h_out,
                          d_out,
                          out_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H (out)");
}

// ---------------------------------------------------------------------
// Explicit cleanup
// Call this once at the very end of training if you want to free GPU
// memory before process exit (otherwise the CUDA runtime will clean up).
// ---------------------------------------------------------------------
extern "C" __declspec(dllexport)
void fedavg_gpu_release()
{
    if (d_clients) { check_cuda(cudaFree(d_clients), "cudaFree(d_clients)"); d_clients = nullptr; }
    if (d_counts)  { check_cuda(cudaFree(d_counts),  "cudaFree(d_counts)");  d_counts  = nullptr; }
    if (d_out)     { check_cuda(cudaFree(d_out),     "cudaFree(d_out)");     d_out     = nullptr; }

    allocated_clients = 0;
    allocated_vec_len = 0;

    std::fprintf(stderr, "[fedavg_gpu] device buffers released.\n");
}
