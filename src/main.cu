#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/01_naive.cuh"
#include "kernels/02_gmem_coalescing.cuh"
#include "kernels/03_smem_blocking.cuh"
#include "kernels/04_1d_blocktiling.cuh"
#include "kernels/05_2d_blocktiling.cuh"
#include "kernels/06_vectorize.cuh"

#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess)                                         \
        {                                                               \
            printf("CUDA ERROR %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                                 \
            exit(1);                                                    \
        }                                                               \
    } while (0)

template <typename F>

float timeMS(F launch, int runs)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
    {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs; // average time per run
}

bool matches(const float *got, const float *ref, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        if (fabsf(got[i] - ref[i]) > 1e-3f * fmaxf(1.0f, fabsf(ref[i])))
        {
            return false;
        }
    }
    return true;
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int sizes[] = {512, 1024, 2048, 4096, 8192};
    printf("%6s %9s %9s %9s %9s %9s %9s %9s %7s %7s %7s %7s %7s %7s %5s %5s %5s %5s %5s %5s\n",
           "N", "naive", "coalesced", "smem", "1D", "2D", "vec", "cuBLAS",
           "naive%", "coal%", "smem%", "1D%", "2D%", "vec%",
           "ok_n", "ok_c", "ok_s", "ok_1", "ok_2", "ok_v");

    /*i didnt know before but its good to have the sweep cuz how it shows how the kernel compares to cuBLAS across sizes,
     since small sizes mostly measure launch overhead and large ones measure real compute, so one size alone mislead .
    */

    for (int N : sizes)
    {
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        float *h_A = (float *)malloc(bytes), *h_B = (float *)malloc(bytes);
        float *h_ref = (float *)malloc(bytes);

        float *h_naive = (float *)malloc(bytes);
        float *h_Coal = (float *)malloc(bytes);
        float *h_smem = (float *)malloc(bytes);
        float *h_1D = (float *)malloc(bytes);
        float *h_2D = (float *)malloc(bytes);
        float *h_vec = (float *)malloc(bytes);

        for (size_t i = 0; i < count; i++)
        {
            h_A[i] = rand() / float(RAND_MAX);
            h_B[i] = rand() / float(RAND_MAX);
        }

        float *d_A, *d_B, *d_Coal, *d_ref, *d_naive, *d_smem, *d_1D, *d_2D, *d_vec;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_ref, bytes));
        CUDA_CHECK(cudaMalloc(&d_Coal, bytes));
        CUDA_CHECK(cudaMalloc(&d_naive, bytes));
        CUDA_CHECK(cudaMalloc(&d_smem, bytes));
        CUDA_CHECK(cudaMalloc(&d_1D, bytes));
        CUDA_CHECK(cudaMalloc(&d_2D, bytes));
        CUDA_CHECK(cudaMalloc(&d_vec, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_Coal, 0, bytes));
        CUDA_CHECK(cudaMemset(d_smem, 0, bytes));
        CUDA_CHECK(cudaMemset(d_naive, 0, bytes));
        CUDA_CHECK(cudaMemset(d_1D, 0, bytes));
        CUDA_CHECK(cudaMemset(d_2D, 0, bytes));
        CUDA_CHECK(cudaMemset(d_vec, 0, bytes));

        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);

        constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
        dim3 block1D((BM * BN) / TM);
        dim3 grid1D((N + BN - 1) / BN, (N + BM - 1) / BM);

        constexpr int BM2 = 128, BN2 = 128, BK2 = 8, TM2 = 8, TN2 = 8;
        dim3 block2D((BM2 * BN2) / (TM2 * TN2));
        dim3 grid2D((N + BN2 - 1) / BN2, (N + BM2 - 1) / BM2);

        auto coal = [&]()
        {
            matmul_coalesced<<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_Coal);
        };

        auto naive = [&]()
        {
            matmul_naive<<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_naive);
        };

        auto smem = [&]()
        {
            matmul_smem<<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_smem);
        };

        auto ref = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_B, N, d_A, N, &beta, d_ref, N);
        };

        auto oneD = [&]()
        {
            matmul_1D<BM, BN, BK, TM><<<grid1D, block1D>>>(N, N, N, alpha, d_A, d_B, beta, d_1D);
        };

        auto twoD = [&]()
        {
            matmul_2D<BM2, BN2, BK2, TM2, TN2><<<grid2D, block2D>>>(N, N, N, alpha, d_A, d_B, beta, d_2D);
        };

        auto vec = [&]()
        {
            matmul_vec<BM2, BN2, BK2, TM2, TN2><<<grid2D, block2D>>>(N, N, N, alpha, d_A, d_B, beta, d_vec);
        };

        int runs = (N >= 2048) ? 3 : 10;

        float msNaive = timeMS(naive, runs);
        float msCoalesced = timeMS(coal, runs);
        float msSmem = timeMS(smem, runs);
        float ms1D = timeMS(oneD, runs);
        float ms2D = timeMS(twoD, runs);
        float msVec = timeMS(vec, runs);

        float msRef = timeMS(ref, runs);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_naive, d_naive, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Coal, d_Coal, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smem, d_smem, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_1D, d_1D, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_2D, d_2D, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vec, d_vec, bytes, cudaMemcpyDeviceToHost));

        bool okNaive = matches(h_naive, h_ref, count);
        bool okCoalesced = matches(h_Coal, h_ref, count);
        bool okSmem = matches(h_smem, h_ref, count);
        bool ok1D = matches(h_1D, h_ref, count);
        bool ok2D = matches(h_2D, h_ref, count);
        bool okVec = matches(h_vec, h_ref, count);

        double flops = 2.0 * N * N * N;
        double gRef = flops / (msRef * 1e6);
        double gNaive = flops / (msNaive * 1e6);
        double gCoalesced = flops / (msCoalesced * 1e6);
        double gSmem = flops / (msSmem * 1e6);
        double g1D = flops / (ms1D * 1e6);
        double g2D = flops / (ms2D * 1e6);
        double gVec = flops / (msVec * 1e6);

        printf("%6d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %5s %5s %5s %5s %5s %5s\n",
               N, gNaive, gCoalesced, gSmem, g1D, g2D, gVec, gRef,
               100.0 * gNaive / gRef, 100.0 * gCoalesced / gRef, 100.0 * gSmem / gRef,
               100.0 * g1D / gRef, 100.0 * g2D / gRef, 100.0 * gVec / gRef,
               okNaive ? "YES" : "NO", okCoalesced ? "YES" : "NO", okSmem ? "YES" : "NO",
               ok1D ? "YES" : "NO", ok2D ? "YES" : "NO", okVec ? "YES" : "NO");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Coal);
        cudaFree(d_ref);
        free(h_A);
        free(h_B);
        free(h_Coal);
        free(h_ref);
        cudaFree(d_naive);
        free(h_naive);
        cudaFree(d_smem);
        free(h_smem);
        cudaFree(d_1D);
        free(h_1D);
        cudaFree(d_2D);
        free(h_2D);
        cudaFree(d_vec);
        free(h_vec);
    }

    cublasDestroy(handle);
    return 0;
}