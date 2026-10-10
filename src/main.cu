#include <cstdio> 
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/01_naive.cuh" 
#include "kernels/02_gmem_coalescing.cuh"

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
      printf("CUDA ERROR %s at %s:%d\n", cudaGetErrorString(err),      \
             __FILE__, __LINE__);                                      \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

template <typename F>

float timeMS(F launch, int runs) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch(); 
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
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

bool matches (const float* got, const float* ref, size_t count) { 
    for (size_t i = 0; i < count; i++) { 
        if (fabsf(got[i] - ref[i]) > 1e-3f * fmaxf(1.0f, fabsf(ref[i]))) { 
            return false; 
        }
    }
    return true; 
}


int main() { 
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f; 
    const float beta = 0.0f;

    int sizes[] = { 512, 1024, 2048, 4096, 8192 };
    printf("%6s %10s %10s %10s %8s %8s %6s %6s\n",
       "N", "naive", "coalesced", "cuBLAS", "naive%", "coal%", "ok_n", "ok_c");

    /*i didnt know before but its good to have the sweep cuz how it shows how the kernel compares to cuBLAS across sizes,
     since small sizes mostly measure launch overhead and large ones measure real compute, so one size alone mislead . 
    */

    for (int N : sizes) { 
        size_t count = (size_t)N*N; 
        size_t bytes = count * sizeof(float); 


        float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes); 
        float *h_C = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes); // h_ref is the cUBLAS reference result
        float *h_naive = (float*)malloc(bytes); // h_naive is the naive kernel result

        for (size_t i = 0; i < count; i++) { 
            h_A[i] = rand() / float(RAND_MAX);
            h_B[i] = rand() / float(RAND_MAX);
        }


        float *d_A, *d_B, *d_C, *d_ref, *d_naive; 
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMalloc(&d_ref, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));


        CUDA_CHECK(cudaMalloc(&d_naive, bytes));
        CUDA_CHECK(cudaMemset(d_naive, 0, bytes));

        dim3 block(32,32);  
        dim3 grid((N+31)/ 32, (N+31)/32); 

        auto newest = [&] () { 
            matmul_coalesced<<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_C);
        }; 

        auto naive = [&] () { 
            matmul_naive<<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_naive);
        };

        auto ref = [&] () { 
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, d_B, N, d_A, N, &beta, d_ref, N);
        };

        int runs = (N >= 2048) ? 3 : 10; 
        float msNewest = timeMS(newest, runs); 
        float msNaive = timeMS(naive, runs);
        float msRef = timeMS(ref, runs); 
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_naive, d_naive, bytes, cudaMemcpyDeviceToHost));
        
        bool okNaive = matches(h_naive, h_ref, count);
        bool okNewest = matches(h_C, h_ref, count);

        double flops = 2.0 * N * N * N; 
        double gNewest = flops / (msNewest * 1e6);
        double gRef = flops / (msRef * 1e6);
        double gNaive = flops / (msNaive * 1e6);

        printf("%6d %10.1f %10.1f %10.1f %7.1f%% %7.1f%% %6s %6s\n",
            N, gNaive, gNewest, gRef,
            100.0 * gNaive / gRef, 100.0 * gNewest / gRef,
            okNaive ? "YES" : "NO", okNewest ? "YES" : "NO");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_ref);
        free(h_A); free(h_B); free(h_C); free(h_ref);
        cudaFree(d_naive); free(h_naive);


    }

    cublasDestroy(handle);
    return 0;

}