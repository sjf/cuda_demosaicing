/*
 * cuda.c
 *
 *  Created on: 9 Feb 2010
 *      Author: sjf65
 */

#include <cutil.h>
#include <cutil_inline.h>
#include <stdlib.h>
#include <assert.h>
#include <driver_types.h>
#include "base.h"
#include "image.h"
#include "cuda_utils.h"
#include "util.h"

void show_mem_info(){
    size_t free = 0, total = 0;
    call(cudaMemGetInfo(&free,&total));
    Info("Total: %0.3fMB Free: %0.3fMB", MB(total), MB(free));
}

void cudaInit() {
    cutilSafeCall(cudaSetDevice( cutGetMaxGflopsDeviceId()));
}

void *devMalloc(size_t size) {
    //show_mem_info();
    //Info("Requesting %0.3fMB",MB(size));

    void *result = NULL;
    call(cudaMalloc(&result, size));
    assert(result);
    return result;
}

void setupTexture(struct textureReference tex) {
    tex.normalized = 0;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
}


void _sync_kern(const char *file, const int line, const char * name){
    // check kernel launch for error
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        FatalError("%s:%i CUDA error: %s : %s.\n",
                   file, line, name, cudaGetErrorString(err));
    }
    err = cudaThreadSynchronize();
    if( cudaSuccess != err) {
        FatalError("%s:%i cudaThreadSynchronize error: %s.\n",
                   file, line, cudaGetErrorString(err));
    }
}

void _call(cudaError err, const char *file, const int line ) {
    if( cudaSuccess != err) {
        print_stack_trace();
        FatalError("%s:%i Runtime API error: %s.\n",
                   file, line, cudaGetErrorString(err) );
    }
}

void _cutcall(CUTBoolean err, const char *file, const int line ) {
    if( CUTTrue != err) {
        print_stack_trace();
        FatalError("%s(%i) : CUTIL CUDA error.\n", file, line);
    }
}



void *memcpy_d_to_h(void *d_ptr, size_t size) {
    void* h_buf = malloc(size);
    call(cudaMemcpy(h_buf, d_ptr, size, cudaMemcpyDeviceToHost));
    return h_buf;
}

void write_d_to_file(pixel *d_ptr, uint width, uint height, char *filename) {
    uint bufsize = width * height * RGB * sizeof(pixel);
    pixel* buf = (pixel *)memcpy_d_to_h(d_ptr,bufsize);
    write_image_to_file(buf,width,height,filename);
    free(buf);
}

void write_d4_to_file(char4 *d_ptr, uint width, uint height, char *filename) {
    uint bufsize = width * height * sizeof(char4);
    char4* buf4 = (char4 *)memcpy_d_to_h(d_ptr,bufsize);

    pixel* buf = (pixel *)malloc(width * height * RGB *sizeof(pixel));

    for (uint i = 0; i < height*width;i++){
        char4 val = buf4[i];
        //Info("RGB: %d,%d,%d",val.x,val.y,val.x);
        buf[i*3+R] = val.x;
        buf[i*3+G] = val.y;
        buf[i*3+B] = val.z;
    }
    write_image_to_file(buf,width,height,filename);
    free(buf);
    free(buf4);
}

void save_d_map(float *d_buf, uint width, uint height, int max_val, const char *filename){
    size_t d_bufsize = height * width * sizeof(float);
    float *buf = (float *)memcpy_d_to_h(d_buf,d_bufsize);

    pixel *image = mallocz<pixel>(height*width*RGB*sizeof(char));
    for (uint c = 0; c < width*height; c++){
        float valf = buf[c];
        uchar val = clampc(round((valf/max_val)*255));
        image[3*c] = val;
        image[3*c+1] = val;
        image[3*c+2] = val;
    }
    write_image_to_file(image,width,height,filename);
    free(buf);
    free(image);
}
void save_d_map_uchar(uchar *d_buf, uint width, uint height, int max_val, const char *filename){
    size_t d_bufsize = height * width * sizeof(uchar);
    uchar *buf = (uchar *)memcpy_d_to_h(d_buf,d_bufsize);

    pixel *image = mallocz<pixel>(height*width*RGB*sizeof(char));
    for (uint c = 0; c < width*height; c++){
        uchar valc = buf[c];
        uchar val = clampc(round(((float)valc/(float)max_val)*255));
        image[3*c] = val;
        image[3*c+1] = val;
        image[3*c+2] = val;
    }
    write_image_to_file(image,width,height,filename);
    free(buf);
    free(image);
}

void dev_quick_show_timer(char *name, float value) {
    const char *unit = "ms";
#ifndef _TEST
    if (value > 1000) {
        value /= 1000;
        unit = "s";
    }
#endif
    Info("CUDA Timer %s: %0.3f (%s)", name, value, unit);
}
