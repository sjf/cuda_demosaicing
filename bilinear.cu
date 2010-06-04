#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>
#include <base.h>
#include <image.h>
#include <bayer.h>
#include <util.h>
#include "cuda_utils.h"

texture<pixel, 2, cudaReadModeElementType> tex;

KERNEL void bilinear_kernel(pixel* g_odata, int width, int height);
KERNEL void nop_kernel(pixel* g_result, int width, int height);

extern "C"
void cuda_bilinear(img *image) {
    Info("Performing CUDA bilinear interpolation");
    uint height = image->height;
    uint width = image->width;

    uint buf_size = width * height * RGB * sizeof(pixel);

    /* Setup raw image array */
    cudaChannelFormatDesc input_channel = cudaCreateChannelDesc<pixel>();
    cudaArray *d_raw_image = NULL;
    cutilSafeCall(cudaMallocArray(&d_raw_image, &input_channel, width * RGB, height));
    assert(d_raw_image);
    startTimer(memCpy);
    cutilSafeCall(cudaMemcpyToArray(d_raw_image, 0, 0,
                     image->buffer, buf_size, cudaMemcpyHostToDevice));
    quickStop(memCpy);
    /* Setup texture */
    startTimer(texBin);
    cutilSafeCall(cudaBindTextureToArray(tex, d_raw_image));

    tex.normalized = 0;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    quickStop(texBin);
    //CUresult cuTexRefSetAddressMode (CUtexref hTexRef, int Dim, CUaddress_mode am)
    //cuTexRefSetAddressMode(tex,0,CU_TR_ADDRESS_MODE_MIRROR);
    //cuTexRefSetAddressMode(tex,1,CU_TR_ADDRESS_MODE_MIRROR);
    //cuTexRefSetAddressMode(tex,2,CU_TR_ADDRESS_MODE_MIRROR);

    /* Setup output */
    startTimer(cudaMalloc);
    pixel *d_result = NULL;
    cutilSafeCall(cudaMalloc((void **)&d_result, buf_size));
    quickStop(cudaMalloc);
    dim3 thread_block(8, 8);
    dim3 block_grid((width + thread_block.x - 1) / thread_block.x,
                 (height + thread_block.y - 1) / thread_block.y);
    //dim3 thread_block(1,1);
    //dim3 block_grid(width, height);

#ifdef _BENCH
    Info("Warm up");
    nop_kernel<<< block_grid, thread_block>>>(d_result, width, height);
    cutilCheckMsg("Kernel execution failed");
#endif

    cutilSafeCall(cudaThreadSynchronize());
    //unsigned int timer = 0;
    //cutilCheckError(cutCreateTimer(&timer));
    //cutilCheckError(cutStartTimer(timer));
    startTimer(kern);
    //Info("Running test");
    bilinear_kernel<<< block_grid, thread_block>>>(d_result, width, height);
    cutilCheckMsg("Kernel execution failed");
    quickStop(kern);
    //(cudaThreadSynchronize());
    //cutilCheckError(cutStopTimer(timer));

    //printf("%d, %d, %f\n", image->width, image->height, cutGetTimerValue(timer));
    //Info("Processing time: %f (ms)", cutGetTimerValue(timer));
    //Info("%.2f Mpixels/sec", (width * height / (cutGetTimerValue(timer) / 1000.0f)) / 1e6);
    //cutilCheckError( cutDeleteTimer( timer));

    /* Copy result from device */
    startTimer(memcpy2);
    cutilSafeCall(
        cudaMemcpy( image->buffer, d_result, buf_size, cudaMemcpyDeviceToHost));
    quickStop(memcpy2);
    startTimer(cufree);
    cutilSafeCall(cudaFree(d_result));
    cutilSafeCall(cudaFreeArray(d_raw_image));
    quickStop(cufree);
    cudaThreadExit();
}

#define tex_get_comp(tex,x,y,c) tex2D((tex),(mirror((x),width))*3+(c),(mirror((y),height)))
//#define tex_get_comp(tex,x,y,c) tex2D((tex),(x)*3+(c),(y))

KERNEL void bilinear_kernel(pixel* g_result, int width, int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int filter_color = get_filter_color(x,y);

    float sum = 0;

    int res_index = (y*width + x)*3;
    pixel *res_pix = g_result+res_index;

    /* Copy existing val to output */
    res_pix[filter_color] = tex_get_comp(tex,x,y,filter_color);

    if (filter_color == R || filter_color == B) {
        /* Red/Blue only pixels */
        /* Green channel */
        sum = tex_get_comp(tex,x-1,y,G) +
              tex_get_comp(tex,x+1,y,G) +
              tex_get_comp(tex,x,y-1,G) +
              tex_get_comp(tex,x,y+1,G);
        res_pix[G] = (uchar)(sum / 4);

        int dest_color = (filter_color == R) ? B : R;
        /* Red/Blue channel */
        sum = tex_get_comp(tex,x-1,y-1,dest_color) +
              tex_get_comp(tex,x-1,y+1,dest_color) +
              tex_get_comp(tex,x+1,y-1,dest_color) +
              tex_get_comp(tex,x+1,y+1,dest_color);
        res_pix[dest_color] = (uchar)(sum / 4);
    } else {
        /* Green only pixels */
        /* Red channel */
        if (even(y)) {
            sum = tex_get_comp(tex,x-1,y,R) +
                  tex_get_comp(tex,x+1,y,R);
            res_pix[R] = (uchar)(sum / 2);

            sum = tex_get_comp(tex,x,y-1,B) +
                  tex_get_comp(tex,x,y+1,B);
            res_pix[B] = (uchar)(sum / 2);
        } else {
            sum = tex_get_comp(tex,x,y-1,R) +
                  tex_get_comp(tex,x,y+1,R);
            res_pix[R] = (uchar)(sum / 2);

            sum = tex_get_comp(tex,x-1,y,B) +
                  tex_get_comp(tex,x+1,y,B);
            res_pix[B] = (uchar)(sum / 2);
        }
    }
}

KERNEL void nop_kernel(pixel* g_result, int width, int height)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int index = (y*width + x)*3;
    uint tex_x = x*3;
    uint tex_y = y;
    for (uint c = 0; c < RGB; c++) {
        pixel val = tex2D(tex, tex_x+c, tex_y);
        //printf("index: %03d %02d,%02d:%d %d\n",index,x,y,c,val);
        g_result[index+c] = val;
    }
}


