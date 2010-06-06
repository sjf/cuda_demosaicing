
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>
#include <base.h>
#include <image.h>
#include <bayer.h>
#include "base.h"
#include "image.h"
#include "cuda_utils.h"
#include "ahd_kernels.cu"
#include "util.h"

void run_cuda_ahd(img *image, pixel*result);

#define MBs (1024 * 1024)
#define MIN_AVAIL 50 * MBs

void cuda_ahd(img *image) {
    uint height = image->height;
    uint width = image->width;
    size_t mem_needed;
    size_t avail = 0, total = 0;
    int n = 0;
    size_t res_size = width * height * RGB * sizeof(pixel);
    pixel *result = mallocz<pixel>(res_size);
    call(cudaMemGetInfo(&avail,&total));
    //show_mem_info();
    do {
        n++;
        size_t buf_size = (width+2*P) * (height+2*P)/n * sizeof(char4);
        size_t cie_buf_size = width * height/n * sizeof(float4);
        size_t homo_buf_size = width * height/n * sizeof(uchar);

        mem_needed = cie_buf_size * 2 + homo_buf_size * 2 + buf_size * 2;
        Info("Image size: %fMB Homo bufsize : %fMB CIE bufsize %fMB Mem needed: %fMB",
                MB(buf_size), MB(homo_buf_size), MB(cie_buf_size), MB(mem_needed));
    } while (mem_needed > avail - MIN_AVAIL);

    int h = (height/n/2) * 2; // round to lowest multiple of 2
    img *cropped_img = new_image(h,width);
    int offset = 0, i;
    
    for (i = 0; i < n-1; i++) {
        Info("Iteration %i, offset %d h: %d n: %d",i,offset,h,n);
        cropped_img->buffer = image->buffer + offset;
        run_cuda_ahd(cropped_img, result + offset);
        //memcpy(result + offset, res, cropped_buf_size);
        //free(res);

        offset += h * width * RGB;
    }
    /* Final tile may be slightly different size due to rounding */
    cropped_img->height = height - i*h;
    Info("Final tile height %d",cropped_img->height);
    cropped_img->buffer = image->buffer + offset;
    run_cuda_ahd(cropped_img, result + offset);
    //memcpy(result + offset, res, cropped_img->height * width * RGB *sizeof(pixel));
    //free(res);

    free(image->buffer);
    image->buffer = result;
}

void run_cuda_ahd(img *image, pixel *result) {
#ifdef __CUDAEMU__
    Info("Performing CUDA AHD interpolation (Emulation mode)");
#else
    Info("Performing CUDA AHD interpolation");
#endif

    uint height = image->height;
    uint width = image->width;
#ifndef __CUDAEMU__
    if (width % 32 > 0) {
        FatalError("Width must be a multiple of 32");
    }
#endif
    Info("Width: %d Height %d\n",width,height);
    size_t buf_size = width * height * RGB * sizeof(pixel);

    uint pheight = height + P*2;
    uint pwidth = width + P*2;
    size_t pbuf_size = pwidth * pheight * sizeof(pixel);
    pixel *pimage = pad_image(image->buffer,width,height,P);

    if (settings->save_temps) {
        save_grayscale(pimage,pwidth,pheight,"img/padded_image.ppm");
    }

    /* Make channels */
    cudaChannelFormatDesc pixel_channel = cudaCreateChannelDesc<pixel>();
    cudaChannelFormatDesc pixel4_channel = cudaCreateChannelDesc<pixel4>();
    cudaChannelFormatDesc float4_channel = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc float_channel = cudaCreateChannelDesc<float>();

    /* Setup source image array on device */
    cudaArray *d_src_image = NULL;
    call(cudaMallocArray(&d_src_image, &pixel_channel, pwidth, pheight));
    call(cudaMemcpyToArray(d_src_image, 0, 0,
            pimage, pbuf_size, cudaMemcpyHostToDevice));
    /* Setup source image texture */
    call(cudaBindTextureToArray(src, d_src_image));
    setupTexture(src);

    pixel *d_horz_tmpres = NULL;
    pixel *d_vert_tmpres = NULL;
    if (settings->save_temps) {
        /* these are just for debugging */
        /* they neccessary for the algorithm */
        d_horz_tmpres = (pixel*)devMalloc(buf_size);
        d_vert_tmpres = (pixel*)devMalloc(buf_size);
    }


    size_t dest_pbuf_size = pwidth * pheight * sizeof(pixel4);
    pixel4 *d_horz_g = (pixel4*)devMalloc(dest_pbuf_size);
    pixel4 *d_vert_g = (pixel4*)devMalloc(dest_pbuf_size);
//    size_t dest_pbuf_size = pwidth * pheight * sizeof(pixel) * RGB;
//    pixel *d_horz_g = (pixel*)devMalloc(dest_pbuf_size);
//    pixel *d_vert_g = (pixel*)devMalloc(dest_pbuf_size);

    dim3 thread_block(32, 8);

    dim3 pblock_grid((pwidth  + thread_block.x - 1) / thread_block.x,
                     (pheight + thread_block.y - 1) / thread_block.y);

    dim3 block_grid((width  + thread_block.x - 1) / thread_block.x,
                    (height + thread_block.y - 1) / thread_block.y);

    /*DebugI(pwidth);
    DebugI(pheight);*/

    Info("Interpolating GREEN");
    /* Interpolate horz and vert green */
    RUN_KERNEL(ahd_kernel_interp_g, pblock_grid, thread_block,
               d_horz_g, d_vert_g, pwidth, pheight);

    devFreeArray(d_src_image);

    if (settings->save_temps) {
        write_d4_to_file(d_horz_g,pwidth,pheight,"img/interp_g_horz.ppm");
        write_d4_to_file(d_vert_g,pwidth,pheight,"img/interp_g_vert.ppm");
    }

    assert(pwidth %32 == 0);
    assert(pwidth*sizeof(pixel4) %32 ==0);

    /* Interpolate horz r/b */
    Info("Interpolating Horizontal RED and BLUE");

    size_t cie_bufsize = width * height * sizeof(float4);
    float4 *d_horz_result = (float4*)devMalloc(cie_bufsize);
//    size_t cie_bufsize = width * height * sizeof(float) * LAB;
//    float *d_horz_result = (float*)devMalloc(cie_bufsize);

    size_t offset = 1;
    call(cudaBindTexture2D(&offset,
            src_g, d_horz_g, pixel4_channel, pwidth, pheight, pwidth*sizeof(pixel4)));
    //src_g, d_horz_g, pixel_channel, pwidth*RGB, pheight, pwidth*sizeof(pixel)*RGB));

    assert(offset == 0); // this should always be zero, but check the CUDA manual wasn't lying
    setupTexture(src_g);

    RUN_KERNEL(ahd_kernel_interp_rb ,block_grid, thread_block,
            d_horz_result, d_horz_tmpres,  pwidth, pheight);
    devFree(d_horz_g);

    float4 *d_vert_result = (float4*)devMalloc(cie_bufsize);
    //float *d_vert_result = (float*)devMalloc(cie_bufsize);
    /* Interpolate vert r/b */
    call(cudaBindTexture2D(&offset,
            src_g, d_vert_g, pixel4_channel, pwidth, pheight, pwidth*sizeof(pixel4)));
            /*src_g, d_vert_g, pixel_channel, pwidth*RGB, pheight, pwidth*sizeof(pixel)*RGB));*/


    assert(offset == 0);
    setupTexture(src_g);
    RUN_KERNEL(ahd_kernel_interp_rb ,block_grid, thread_block,
                 d_vert_result, d_vert_tmpres,  pwidth, pheight);
    devFree(d_vert_g);

    if (settings->save_temps && d_horz_tmpres != NULL && d_vert_tmpres != NULL) {
        write_d_to_file(d_horz_tmpres,width,height,"img/interpolation_horz.ppm");
        write_d_to_file(d_vert_tmpres,width,height,"img/interpolation_vert.ppm");
    }

    call(cudaBindTexture2D(NULL, horz_tex, d_horz_result, float4_channel,
            width, height, width*sizeof(float4)));
    call(cudaBindTexture2D(NULL, vert_tex, d_vert_result, float4_channel,
            width, height, width*sizeof(float4)));

//    call(cudaBindTexture2D(NULL, horz_tex, d_horz_result, float_channel,
//            width*RGB, height, width*sizeof(float)*RGB));
//    call(cudaBindTexture2D(NULL, vert_tex, d_vert_result, float_channel,
//            width*RGB, height, width*sizeof(float)*RGB));

    setupTexture(horz_tex);
    setupTexture(vert_tex);

    size_t homo_bufsize = height * width * sizeof(uchar);
    uchar *d_homo_horz = (uchar *)devMalloc(homo_bufsize);
    uchar *d_homo_vert = (uchar *)devMalloc(homo_bufsize);
    RUN_KERNEL(ahd_kernel_build_homo_map, block_grid, thread_block,
            d_homo_horz, d_homo_vert, width, height/*, settings->ball_distance*/);

    if (settings->save_temps) {
        int scale = ball_area(settings->ball_distance);
        save_d_map_uchar(d_homo_horz,width,height,scale,"img/homo_map_horz.ppm");
        save_d_map_uchar(d_homo_vert,width,height,scale,"img/homo_map_vert.ppm");
    }

    call(cudaBindTexture2D(NULL,
            homo_h_tex, d_homo_horz, pixel_channel, width, height, width*sizeof(uchar)));
    call(cudaBindTexture2D(NULL,
            homo_v_tex, d_homo_vert, pixel_channel, width, height, width*sizeof(uchar)));
    setupTexture(homo_h_tex);
    setupTexture(homo_v_tex);

    float *d_direction_tmpres = NULL;
    if (settings->save_temps){
        d_direction_tmpres = (float*)devMalloc(width * height * sizeof(float));
    }
    pixel *d_result = (pixel*)devMalloc(buf_size);

    RUN_KERNEL(ahd_kernel_choose_direction, block_grid, thread_block,
          d_result,d_direction_tmpres,width,height);

    if (settings->save_temps) {
        save_d_map(d_direction_tmpres,width,height,1,"img/direction.ppm");
        write_d_to_file(d_result,width,height,"img/pre_noise.ppm");
    }

    devFree(d_horz_g);
    devFree(d_vert_g);
    devFree(d_horz_result);
    devFree(d_vert_result);

    pixel *d_temp = (pixel*)devMalloc(buf_size);


    for (uint i = 0; i < settings->median_filter_iterations; i++) {
        Info("Removing artefacts");
        call(cudaBindTexture2D(NULL,
                src, d_result, pixel_channel, width*RGB, height, width*RGB*sizeof(pixel)));
        RUN_KERNEL(ahd_kernel_remove_artefacts,block_grid,thread_block,d_temp, width, height);

        pixel *swap = d_result;
        d_result = d_temp; d_temp = swap;
    }
//    if (settings->median_filter_iterations %2) {
//        free(d_temp);
//    } else {
//        free(image->buffer);
//    }


    /* Copy result from device */
    //pixel *result = (pixel *)memcpy_d_to_h(d_temp,buf_size);
    call(cudaMemcpy(result, d_temp, buf_size, cudaMemcpyDeviceToHost));


    devFree(d_result);
    devFree(d_direction_tmpres);
    devFree(d_temp);
    devFreeArray(d_src_image);
    free(pimage);

//    cudaThreadExit();
//    free(image->buffer);
//    image->buffer = result;
//    return result;
}
