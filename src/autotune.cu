#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "common.cuh"
#include "kernels/09_autotuned.cuh"

template <int BM, int BN, int BK, int TM, int TN>
void tryConfig(int N, const float *d_A, const float *d_B, float *d_C,
               const float *h_ref, float *h_C, double flops)
{
    if constexpr (!validConfig<BM, BN, BK, TM, TN>())
    {
        printf("%4d %4d %3d %3d %3d %8s %10s %5s\n", BM, BN, BK, TM, TN, "-", "skipped", "-");
    }
    else
    {
        constexpr int threads = numThreads<BM, BN, TM, TN>();
        const float alpha = 1.0f, beta = 0.0f;
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        dim3 block(threads);
        dim3 grid(N / BN, N / BM);

        CUDA_CHECK(cudaMemset(d_C, 0, bytes));

        auto launch = [&]()
        {
            matmul_tuned<BM, BN, BK, TM, TN><<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_C);
        };

        float ms = timeMS(launch, 5);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        bool ok = matches(h_C, h_ref, count);

        printf("%4d %4d %3d %3d %3d %8d %10.1f %5s\n",
               BM, BN, BK, TM, TN, threads, flops / (ms * 1e6), ok ? "YES" : "NO");
    }
}

#define TRY(BM, BN, BK, TM, TN) tryConfig<BM, BN, BK, TM, TN>(N, d_A, d_B, d_C, h_ref, h_C, flops)

int main()
{
    const int N = 4096;
    size_t count = (size_t)N * N;
    size_t bytes = count * sizeof(float);
    double flops = 2.0 * N * N * N;

    float *h_A = (float *)malloc(bytes), *h_B = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes), *h_C = (float *)malloc(bytes);
    for (size_t i = 0; i < count; i++)
    {
        h_A[i] = rand() / float(RAND_MAX);
        h_B[i] = rand() / float(RAND_MAX);
    }

    float *d_A, *d_B, *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // cuBLAS reference result + its speed
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    auto ref = [&]()
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_ref, N);
    };
    float msRef = timeMS(ref, 5);
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));
    printf("N = %d, cuBLAS: %.1f GFLOPS\n\n", N, flops / (msRef * 1e6));

    printf("%4s %4s %3s %3s %3s %8s %10s %5s\n", "BM", "BN", "BK", "TM", "TN", "threads", "GFLOPS", "ok");

    TRY(128, 128, 8, 8, 8);
    TRY(128, 128, 16, 8, 8);
    TRY(128, 128, 32, 8, 8);
    TRY(64, 64, 16, 4, 4);
    TRY(64, 64, 16, 8, 8); // only 64 threads per block
    TRY(128, 64, 8, 8, 8); // As load needs 2 passes
    TRY(64, 128, 8, 8, 8);
    TRY(128, 128, 16, 8, 4); // 512 threads
    TRY(128, 128, 16, 4, 8); // 512 threads
    TRY(128, 128, 8, 4, 8);  // illegal on purpose: shows up as "skipped"
    TRY(128, 128, 8, 8, 8);  // baseline again

    cublasDestroy(handle);
    return 0;
}