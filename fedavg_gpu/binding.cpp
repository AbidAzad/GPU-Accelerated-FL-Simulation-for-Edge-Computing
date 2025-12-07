#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Declare CUDA functions
extern "C" void fedavg_weighted_average_gpu(
    const float* h_client_weights,
    const int*   h_counts,
    float*       h_out,
    int          num_clients,
    int          vec_len
);

py::array_t<float> fedavg_weighted_average(
    py::array_t<float> flat_stack,   // shape (K, P)
    py::array_t<int> counts           // shape (K,)
) {
    auto buf_w = flat_stack.request();
    auto buf_c = counts.request();

    int K = buf_w.shape[0];
    int P = buf_w.shape[1];

    auto out = py::array_t<float>({P});

    fedavg_weighted_average_gpu(
        (float*)buf_w.ptr,
        (int*)  buf_c.ptr,
        (float*)out.request().ptr,
        K, P
    );

    return out;
}

PYBIND11_MODULE(fedavg_gpu, m) {
    m.def("fedavg_weighted_average", &fedavg_weighted_average,
          "FedAvg weighted average on GPU");
}
