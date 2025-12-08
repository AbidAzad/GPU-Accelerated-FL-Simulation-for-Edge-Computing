// fedavg_gpu_windows.cu — Windows DLL with loud logging and no exceptions

#include <cuda_runtime.h>
#include <windows.h>

#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

// Export macro
#define DLL_EXPORT extern "C" __declspec(dllexport)

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
// client_weights: shape (num_clients, vec_len), row-major
// scales        : shape (num_clients,) (sum to 1)
// out           : shape (vec_len,)
// ------------------------------------------------------------------
__global__ void fedavg_kernel(
    const float* __restrict__ client_weights,
    const float* __restrict__ scales,
    float* __restrict__ out,
    int num_clients,
    int vec_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_len) return;

    float acc = 0.0f;
    // For each parameter index idx, accumulate over clients
    for (int c = 0; c < num_clients; ++c) {
        int offset = c * vec_len + idx;
        acc += scales[c] * client_weights[offset];
    }
    out[idx] = acc;
}

// ------------------------------------------------------------------
// Exported entry point (called from Python via ctypes)
//
// h_client_weights: pointer to host float array of size num_clients * vec_len
// h_counts       : pointer to host int array of size num_clients
// h_out          : pointer to host float array of size vec_len (output)
// ------------------------------------------------------------------
DLL_EXPORT void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len)
{

    if (!h_client_weights || !h_counts || !h_out) {
        std::cerr << "[fedavg_gpu] ERROR: null pointer(s) passed in."
                  << std::endl;
        return;
    }
    if (num_clients <= 0 || vec_len <= 0) {
        return;
    }

    // Compute total count and scales
    long long total_count = 0;
    for (int i = 0; i < num_clients; ++i) {
        total_count += static_cast<long long>(h_counts[i]);
    }

    if (total_count <= 0) {
        std::cerr << "[fedavg_gpu] ERROR: total_count <= 0, aborting."
                  << std::endl;
        return;
    }

    std::vector<float> h_scales(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        h_scales[i] = static_cast<float>(h_counts[i]) /
                      static_cast<float>(total_count);
    }

    // Device allocations
    size_t weights_bytes = static_cast<size_t>(num_clients) *
                           static_cast<size_t>(vec_len) *
                           sizeof(float);
    size_t scales_bytes  = static_cast<size_t>(num_clients) *
                           sizeof(float);
    size_t out_bytes     = static_cast<size_t>(vec_len) *
                           sizeof(float);

    float *d_client_weights = nullptr;
    float *d_scales         = nullptr;
    float *d_out            = nullptr;

    CUDA_CHECK(cudaMalloc(&d_client_weights, weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_client_weights,
                          h_client_weights,
                          weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales,
                          h_scales.data(),
                          scales_bytes,
                          cudaMemcpyHostToDevice));


    // Kernel launch config
    int threadsPerBlock = 256;
    int blocks = (vec_len + threadsPerBlock - 1) / threadsPerBlock;

    fedavg_kernel<<<blocks, threadsPerBlock>>>(
        d_client_weights,
        d_scales,
        d_out,
        num_clients,
        vec_len
    );

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "[fedavg_gpu] KERNEL LAUNCH ERROR: "
                  << cudaGetErrorString(launchErr) << std::endl;
        // Still try to continue to capture as much info as possible.
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out,
                          d_out,
                          out_bytes,
                          cudaMemcpyDeviceToHost));

    cudaFree(d_client_weights);
    cudaFree(d_scales);
    cudaFree(d_out);
}

// Standard DLL entry; no special handling
BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD   ul_reason_for_call,
                      LPVOID  lpReserved)
{
    return TRUE;
}
