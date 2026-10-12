#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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
    return ms / runs;
}

inline bool matches(const float *got, const float *ref, size_t count)
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