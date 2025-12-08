// fedavg_gpu.cu — Linux .so with loud logging and NO C++ exceptions

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstddef>

// ------------------------------------------------------------------
// Logging + safe CUDA macro (NO EXCEPTIONS)
// ------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::cerr << "[fedavg_gpu] CUDA error at " << __FILE__ << ":"    \
                      << __LINE__ << " : " << cudaGetErrorString(_err)       \
                      << std::endl;                                          \
            /* DO NOT THROW. Just bail out. */                               \
            return;                                                           \
        }                                                                     \
    } while (0)

// ------------------------------------------------------------------
// Simple FedAvg kernel
// d_client_weights: shape (num_clients, vec_len), row-major
// d_scales        : shape (num_clients,) (sum to 1)
// d_out           : shape (vec_len,)
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Host entry point (called from Python via ctypes)
//
// h_client_weights: pointer to host float array of size num_clients * vec_len
// h_counts        : pointer to host int array of size num_clients
// h_out           : pointer to host float array of size vec_len (output)
// ------------------------------------------------------------------
extern "C" void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len)
{
    // Basic pointer / shape checks
    if (!h_client_weights || !h_counts || !h_out) {
        std::cerr << "[fedavg_gpu] ERROR: null pointer(s) passed in."
                  << std::endl;
        return;
    }
    if (num_clients <= 0 || vec_len <= 0) {
        std::cerr << "[fedavg_gpu] ERROR: num_clients or vec_len <= 0."
                  << std::endl;
        return;
    }

    // 1) Compute total number of samples on host
    long long total_samples = 0;
    for (int c = 0; c < num_clients; ++c) {
        total_samples += static_cast<long long>(h_counts[c]);
    }
    if (total_samples <= 0) {
        std::cerr << "[fedavg_gpu] ERROR: total_samples <= 0, aborting."
                  << std::endl;
        return;
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

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "[fedavg_gpu] KERNEL LAUNCH ERROR: "
                  << cudaGetErrorString(launch_err) << std::endl;
        // 继续往下走，后面的 CUDA_CHECK(cudaDeviceSynchronize) 会 return 掉
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    // 7) Free device buffers
    cudaFree(d_client_weights);
    cudaFree(d_scales);
    cudaFree(d_out);
}
