#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <iostream>

// Simple CUDA error check macro
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error("CUDA failure");                         \
        }                                                                     \
    } while (0)

/// CUDA kernel: compute weighted FedAvg for one flattened layer.
///
/// Arguments:
///   d_client_weights : device pointer, shape (num_clients * vec_len)
///                      layout: client-major, i.e. client c starts at
///                      offset c * vec_len.
///   d_scales         : device pointer, shape (num_clients),
///                      where d_scales[c] = n_c / total_samples.
///   d_out            : device pointer, shape (vec_len),
///                      output averaged weights.
///   num_clients      : number of participating clients K.
///   vec_len          : length of each client's weight vector L.
///
/// For each index idx in [0, vec_len):
///   out[idx] = sum_c ( d_scales[c] * d_client_weights[c * vec_len + idx] )
///
__global__ void fedavg_kernel(
    const float* __restrict__ d_client_weights,
    const float* __restrict__ d_scales,
    float* __restrict__ d_out,
    int num_clients,
    int vec_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_len) return;

    float acc = 0.0f;

    // Loop over clients and accumulate a weighted sum
    for (int c = 0; c < num_clients; ++c) {
        float w = d_client_weights[c * vec_len + idx];
        float s = d_scales[c];  // = n_c / total_samples
        acc += s * w;
    }

    d_out[idx] = acc;
}

/// Host helper: run FedAvg on GPU for a single flattened layer.
///
/// Parameters:
///   h_client_weights : host pointer, size (num_clients * vec_len)
///   h_counts         : host pointer, size (num_clients)
///   h_out            : host pointer, size (vec_len), output buffer
///   num_clients      : number of clients (K)
///   vec_len          : length of per-client vector (L)
///
/// Notes:
///   - This function allocates temporary device buffers, launches the kernel,
///     and copies the result back to host.
///   - You can later optimize by reusing device buffers across layers / rounds.
///
extern "C" void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len)
{
    if (num_clients <= 0 || vec_len <= 0) {
        throw std::invalid_argument("num_clients and vec_len must be positive");
    }

    // 1) Compute total number of samples on host
    long long total_samples = 0;
    for (int c = 0; c < num_clients; ++c) {
        total_samples += static_cast<long long>(h_counts[c]);
    }
    if (total_samples <= 0) {
        throw std::invalid_argument("total_samples must be positive");
    }

    // 2) Prepare scales on host: s_c = n_c / total_samples
    std::vector<float> h_scales(num_clients);
    for (int c = 0; c < num_clients; ++c) {
        h_scales[c] = static_cast<float>(h_counts[c]) /
                      static_cast<float>(total_samples);
    }

    // 3) Allocate device buffers
    float* d_client_weights = nullptr;
    float* d_scales         = nullptr;
    float* d_out            = nullptr;

    size_t weights_bytes = static_cast<size_t>(num_clients) *
                            static_cast<size_t>(vec_len) * sizeof(float);
    size_t scales_bytes  = static_cast<size_t>(num_clients) * sizeof(float);
    size_t out_bytes     = static_cast<size_t>(vec_len) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_client_weights, weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales,         scales_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,            out_bytes));

    // 4) Copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(d_client_weights, h_client_weights,
                          weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
                          scales_bytes, cudaMemcpyHostToDevice));

    // 5) Launch kernel
    int threads_per_block = 256;
    int blocks = (vec_len + threads_per_block - 1) / threads_per_block;

    fedavg_kernel<<<blocks, threads_per_block>>>(
        d_client_weights, d_scales, d_out, num_clients, vec_len
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    // 7) Free device buffers
    CUDA_CHECK(cudaFree(d_client_weights));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_out));
}
