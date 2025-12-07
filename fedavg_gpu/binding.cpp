#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Declare CUDA 
extern "C" void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len);

// Python wrapper: Input 2D float array 1D int array, output 1D float array
py::array_t<float> fedavg_weighted_average(
    py::array_t<float, py::array::c_style | py::array::forcecast> client_weights,
    py::array_t<int,   py::array::c_style | py::array::forcecast> counts)
{
    auto buf_w = client_weights.request();
    auto buf_c = counts.request();

    if (buf_w.ndim != 2) {
        throw std::runtime_error("client_weights must be 2D");
    }
    if (buf_c.ndim != 1) {
        throw std::runtime_error("counts must be 1D");
    }

    int num_clients = static_cast<int>(buf_w.shape[0]);
    int vec_len     = static_cast<int>(buf_w.shape[1]);

    if (buf_c.shape[0] != num_clients) {
        throw std::runtime_error("counts length must match num_clients");
    }

    const float* h_client_weights = static_cast<float*>(buf_w.ptr);
    const int*   h_counts         = static_cast<int*>(buf_c.ptr);

    // Output 1D vector (vec_len)
    py::array_t<float> out(vec_len);
    auto buf_out = out.request();
    float* h_out = static_cast<float*>(buf_out.ptr);

    fedavg_weighted_average_gpu(
        h_client_weights,
        h_counts,
        h_out,
        num_clients,
        vec_len
    );

    return out;
}

PYBIND11_MODULE(fedavg_gpu, m) {
    m.doc() = "CUDA FedAvg GPU extension";
    m.def("fedavg_weighted_average", &fedavg_weighted_average,
          "Weighted FedAvg on GPU");
}
