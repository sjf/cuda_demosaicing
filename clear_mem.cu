/*
 * clear_mem.cpp
 *
 *  Created on: 26 Apr 2010
 *      Author: sjf
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

__global__ void zero_kernel(int* g_odata, int width, int height);

int main(int argc, char **argv) {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free,&total);
    printf("Total: %0.3fMB Free: %0.3fMB\n", (total)/1024.0/1024.0, (free)/1024.0/1024.0);

    size_t buf_size = free - 100*1024*1024;
    int n = buf_size / sizeof(int);
    int w = n / 1024;
    int h = 1024;
    int *d_mem = NULL;
    cutilSafeCall(cudaMalloc((void **)&d_mem, buf_size));
    dim3 thread_block(8, 8);
    dim3 block_grid((w + thread_block.x - 1) / thread_block.x,
                 (h + thread_block.y - 1) / thread_block.y);

    cutilSafeCall(cudaThreadSynchronize());

    //Info("Running test");
    zero_kernel<<< block_grid, thread_block>>>(d_mem,w,h);
    cutilCheckMsg("Kernel execution failed");

    cudaThreadExit();
}

__global__ void zero_kernel(int* g_result, int width, int height)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int index = (y*width + x);
    g_result[index] = 0;
}
