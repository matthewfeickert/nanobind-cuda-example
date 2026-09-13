// gpu-pairwise — the full Euclidean distance matrix between two point sets,
// computed on the GPU and handed back to NumPy.
//
// One CUDA thread per output element D[i, j] = || X[i, :] - Y[j, :] ||_2.
// This is the naive O(n * m * d) kernel; it exists to show the packaging
// story, not to compete with cuBLAS-based formulations.

#include <cuda_runtime.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

// RAII wrapper so that a thrown exception cannot leak device memory.
template <typename T> struct DeviceBuffer {
    T *ptr = nullptr;
    explicit DeviceBuffer(size_t n) { cuda_check(cudaMalloc(&ptr, n * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

template <typename T>
__global__ void pairwise_kernel(const T *__restrict__ x, const T *__restrict__ y,
                                T *__restrict__ out, int n, int m, int d)
{
    // 2D grid: threadIdx/blockIdx.x walks Y (columns), .y walks X (rows).
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= m) return;

    T acc = 0;
    for (int k = 0; k < d; ++k) {
        T diff = x[i * d + k] - y[j * d + k];
        acc += diff * diff;
    }
    out[i * m + j] = sqrt(acc);
}

// Inputs are C-contiguous (n, d) and (m, d) host arrays with matching dtype;
// output is a freshly allocated (n, m) host array of the same dtype.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>>
pairwise_distances(nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu> x,
                   nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu> y)
{
    if (x.shape(1) != y.shape(1)) {
        throw std::invalid_argument("x and y must have the same number of columns (features)");
    }
    const size_t n = x.shape(0), m = y.shape(0), d = x.shape(1);

    DeviceBuffer<T> dx(n * d), dy(m * d), dout(n * m);
    cuda_check(cudaMemcpy(dx.ptr, x.data(), n * d * sizeof(T), cudaMemcpyHostToDevice), "copy x");
    cuda_check(cudaMemcpy(dy.ptr, y.data(), m * d * sizeof(T), cudaMemcpyHostToDevice), "copy y");

    const dim3 block(32, 8);
    const dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    pairwise_kernel<T><<<grid, block>>>(dx.ptr, dy.ptr, dout.ptr, static_cast<int>(n),
                                        static_cast<int>(m), static_cast<int>(d));
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel execution");

    // Hand ownership of the host result to NumPy via a capsule deleter.
    T *result = new T[n * m];
    cuda_check(cudaMemcpy(result, dout.ptr, n * m * sizeof(T), cudaMemcpyDeviceToHost), "copy result");
    nb::capsule owner(result, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(result, {n, m}, owner);
}

}  // namespace

NB_MODULE(_core, m)
{
    m.doc() = "CUDA pairwise Euclidean distances, exposed through nanobind";
    // Two overloads under one name; nanobind dispatches on the array dtype.
    m.def("pairwise_distances", &pairwise_distances<float>, nb::arg("x"), nb::arg("y"));
    m.def("pairwise_distances", &pairwise_distances<double>, nb::arg("x"), nb::arg("y"));

    m.def("device_name", []() {
        int dev = 0;
        cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        return std::string(prop.name);
    });
}
