// This is a part of qsbd
/**
@file /sbd/framework/cuda_reduce.h
@brief function for vector on distributed-memory
*/

#ifndef SBD_FRAMEWORK_REDUCE_CUDA_H
#define SBD_FRAMEWORK_REDUCE_CUDA_H

#include <cuda.h>

#define _WS 32
#define _MAX_THD 1024

namespace sbd {

template <typename kernel_t>
__global__ void dev_precise_reduce_with_function(double *pReduceBuffer, kernel_t func, size_t count)
{
    __shared__ double cache[_MAX_THD * 2 / _WS];
    double sum, t, v;
    double c = 0.0;
    size_t i, j, nw;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count)
        sum = 0.0;
    else
        sum = func(i);

    // reduce in warp
    nw = min(blockDim.x, _WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor_sync(0xffffffff, c, j, 32);
        v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
        t = sum + v;
        c = (t - sum) - v;
        sum = t;
    }

    if (blockDim.x > _WS) {
        // reduce in thread block
        if ((threadIdx.x & (_WS - 1)) == 0) {
            cache[(threadIdx.x / _WS) * 2] = sum;
            cache[(threadIdx.x / _WS) * 2 + 1] = c;
        }
        __syncthreads();
        if (threadIdx.x < _WS) {
            if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS)) {
                sum = cache[threadIdx.x*2];
                c = cache[threadIdx.x*2 + 1];
            } else {
                sum = 0.0;
                c = 0.0;
            }

            // reduce in warp
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c += __shfl_xor_sync(0xffffffff, c, j, 32);
                v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
                t = sum + v;
                c = (t - sum) - v;
                sum = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        pReduceBuffer[blockIdx.x * 2] = sum;
        pReduceBuffer[blockIdx.x * 2 + 1] = c;
    }
}

__global__ void dev_precise_reduce(double *pReduceBuffer, size_t count)
{
    __shared__ double cache[_MAX_THD * 2 / _WS];
    double sum, t, v;
    double c = 0.0;
    size_t i, j, nw;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count)
        sum = 0.0;
    else{
        sum = pReduceBuffer[i*2];
        c = pReduceBuffer[i*2 + 1];
    }

    // reduce in warp
    nw = min(blockDim.x, _WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor_sync(0xffffffff, c, j, 32);
        v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
        t = sum + v;
        c = (t - sum) - v;
        sum = t;
    }

    if (blockDim.x > _WS) {
        // reduce in thread block
        if ((threadIdx.x & (_WS - 1)) == 0) {
            cache[(threadIdx.x / _WS) * 2] = sum;
            cache[(threadIdx.x / _WS) * 2 + 1] = c;
        }
        __syncthreads();
        if (threadIdx.x < _WS) {
            if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS)) {
                sum = cache[threadIdx.x*2];
                c = cache[threadIdx.x*2 + 1];
            } else {
                sum = 0.0;
                c = 0.0;
            }

            // reduce in warp
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c += __shfl_xor_sync(0xffffffff, c, j, 32);
                v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
                t = sum + v;
                c = (t - sum) - v;
                sum = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        pReduceBuffer[blockIdx.x * 2] = sum;
        pReduceBuffer[blockIdx.x * 2 + 1] = c;
    }
}

template <typename Function>
double precise_reduce_sum_with_function(Function func, size_t size)
{
    size_t n, nt, nb;
    nb = 1;
    nt = size;

    if (nt > _MAX_THD) {
        nb = (nt + _MAX_THD - 1) / _MAX_THD;
        nt = _MAX_THD;
    }
    double* buf;
    cudaMalloc(&buf, nb * 2 * sizeof(double));

    dev_precise_reduce_with_function<Function>
                <<<nb, nt>>>(buf, func, size);
    while (nb > 1) {
        n = nb;
        nt = nb;
        nb = 1;
        if (nt > _MAX_THD) {
            nb = (nt + _MAX_THD - 1) / _MAX_THD;
            nt = _MAX_THD;
        }
        dev_precise_reduce<<<nb, nt>>>(buf, n);
    }

    double ret;
    cudaMemcpy(&ret, buf, sizeof(double), cudaMemcpyDeviceToHost);
    double c;
    cudaMemcpy(&c, buf+1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(buf);

    return ret;
}


}

#endif
