/*
 * cuda_utils.h
 *
 *  Created on: 9 Feb 2010
 *      Author: sjf65
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_
#include <driver_types.h>
#include <cutil.h>
#include <vector_types.h>

#ifdef __CUDACC__
#define KERNEL __global__
#else
#define KERNEL
#endif

#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

// Avoid syntax errors in eclipse
#ifdef __CUDACC__
#define EXEC(g,b) <<<(g),(b)>>>
#else
#define EXEC(g,b)
#endif

#define cutcall(err) _cutcall(err, __FILE__, __LINE__)
#define call(err)  _call(err,__FILE__,__LINE__)
#define thread_sync(name)  _sync_kern(__FILE__,__LINE__,name)

#define devFree(ptr)   do { if (ptr != NULL) call(cudaFree(ptr)); ptr = NULL; } while(0)
#define devFreeArray(ptr)   do { if (ptr != NULL) call(cudaFreeArray(ptr)); ptr = NULL; } while(0)

#define devStartTimer(kern,n) unsigned int CONCAT(cutimer_##kern,n) = 0; \
                              cutcall(cutCreateTimer(CONCAT(&cutimer_##kern,n))); \
                              cutcall(cutStartTimer(CONCAT(cutimer_##kern,n)));

#define devStopTimer(kern,n)  thread_sync(); \
                              cutcall(cutStopTimer(CONCAT(cutimer_##kern,n))); \
                              dev_quick_show_timer(#kern, cutGetTimerValue(CONCAT(cutimer_##kern,n))); \
                              cutcall(cutDeleteTimer(CONCAT(cutimer_##kern,n)));

#define CONCAT( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT( x, y )


#ifdef _TIME_CUDA
#define RUN_KERNEL(kern,grid,block,args...) _TIME_KERNEL(__LINE__,kern,grid,block,args)
#else
#define RUN_KERNEL(kern,grid,block,args...)  kern EXEC((grid),(block)) (args); \
                                             thread_sync(#kern);
#endif

#define _TIME_KERNEL(n,kern,grid,block,args...)  thread_sync(); \
                                                 devStartTimer(kern,n); \
                                                 kern EXEC((grid),(block)) (args); \
                                                 devStopTimer(kern,n);

void cudaInit();
void *devMalloc(size_t size);
void *memcpy_d_to_h(void *d_ptr, size_t size);
void write_d_to_file(pixel *d_ptr, uint width, uint height, char *filename);
void write_d4_to_file(char4 *d_ptr, uint width, uint height, char *filename);
void _call(cudaError err, const char *file, const int line );
void _cutcall(CUTBoolean err, const char *file, const int line );
void _sync_kern(const char *file, const int line, const char *name);
void setupTexture(struct textureReference tex);
void save_d_map(float *d_buff, uint width, uint height, int max_val, const char *filename);
void save_d_map_uchar(uchar *d_buf, uint width, uint height, int max_val, const char *filename);
void dev_quick_show_timer(char * n, float value);
void show_mem_info();

#endif /* CUDA_UTILS_H_ */
