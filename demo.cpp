
#include <iostream>
#include <vector>
#include <cmath>

// Declare the GPU function implemented in fedavg_gpu.cu
extern "C" void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len);

// Simple CPU reference implementation for comparison
void fedavg_weighted_average_cpu(
    const std::vector<float>& client_weights, // shape: (K * L)
    const std::vector<int>&   counts,         // shape: (K)
    std::vector<float>&       out,            // shape: (L)
    int                       num_clients,
    int                       vec_len)
{
    long long total_samples = 0;
    for (int c = 0; c < num_clients; ++c) {
        total_samples += static_cast<long long>(counts[c]);
    }

    for (int i = 0; i < vec_len; ++i) {
        double acc = 0.0;
        for (int c = 0; c < num_clients; ++c) {
            int offset = c * vec_len + i;
            acc += static_cast<double>(counts[c]) *
                   static_cast<double>(client_weights[offset]);
        }
        out[i] = static_cast<float>(acc / static_cast<double>(total_samples));
    }
}

int main() {
    const int num_clients = 2;
    const int vec_len     = 4;

    // Layout: [client0[0..3], client1[0..3]]
    std::vector<float> h_client_weights = {
        // client 0
        1.0f, 2.0f, 3.0f, 4.0f,
        // client 1
        5.0f, 6.0f, 7.0f, 8.0f
    };

    std::vector<int> h_counts = {10, 30};   // n0=10, n1=30

    std::vector<float> h_out_gpu(vec_len, 0.0f);
    std::vector<float> h_out_cpu(vec_len, 0.0f);

    // Compute on CPU for reference
    fedavg_weighted_average_cpu(
        h_client_weights, h_counts, h_out_cpu,
        num_clients, vec_len
    );

    // Compute on GPU
    fedavg_weighted_average_gpu(
        h_client_weights.data(),
        h_counts.data(),
        h_out_gpu.data(),
        num_clients,
        vec_len
    );

    // Print results
    std::cout << "CPU result: ";
    for (int i = 0; i < vec_len; ++i) {
        std::cout << h_out_cpu[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "GPU result: ";
    for (int i = 0; i < vec_len; ++i) {
        std::cout << h_out_gpu[i] << " ";
    }
    std::cout << std::endl;

    // Check difference
    bool ok = true;
    for (int i = 0; i < vec_len; ++i) {
        float diff = std::fabs(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > 1e-5f) {
            ok = false;
            std::cout << "Mismatch at index " << i
                      << " diff=" << diff << std::endl;
        }
    }

    if (ok) {
        std::cout << "FedAvg GPU matches CPU reference." << std::endl;
    } else {
        std::cout << "FedAvg GPU does NOT match CPU reference." << std::endl;
    }

    return 0;
}
