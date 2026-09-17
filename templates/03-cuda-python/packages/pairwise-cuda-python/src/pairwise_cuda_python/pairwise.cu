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
