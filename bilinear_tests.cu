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

img *read_image_from_file(char *filename){
    img *result = NULL;

    FILE *fh = fopen(filename,"r");
    if (!fh) {
        FatalSysError("Could not open '%s'", filename);
    }
    char buf[1024];
    if (fscanf(fh,"%1024s",buf) < 1) {
        FatalError("'%s' is not a valid PPM file", filename);
    }
    if (strcmp(buf,"P6") != 0) {
        FatalError("'%s' is not a PPM file", filename);
    }
    /* Read comments */
    int res = fscanf(fh," #%1024[^\n\r]",buf);
    if (res < 0) {
        FatalError("Error reading PPM file '%s'",filename);
    }

    int height, width, depth;
    height = width = depth = -1;

    /* width, height, depth followed by exactly 1 whitespace character */
    res = fscanf(fh,"%d %d %d%1[\n\t ]",&width,&height,&depth,buf);
    if (res < 0) {
        FatalSysError("Error reading width, height, depth from PPM file '%s'", filename);
    }
    if (res < 3) {
        FatalError("Could not read width, height, depth from PPM file '%s'", filename);
    }
    if (depth <= 0 || depth > 255) {
        FatalError("Error reading PPM file '%s', invalid color depth",filename);
    }
    //uint pix_width = (depth <= 255) ? 1 : 2;
    if (depth > 255) {
        FatalError("Colour depth %d not supported", depth);
    }
    Info("Loading PPM image %dx%d depth %d",width,height,depth);

    int buf_size = height*width*RGB;
    uchar *buffer = mallocz<uchar>(buf_size);
    res = fread(buffer,sizeof(uchar),buf_size,fh);
    if (res < 0) {
        FatalSysError("Error reading PPM file '%s'",filename);
    }
    if (res < buf_size) {
        FatalError("Error reading PPM file '%s', file too short, read %d bytes, should be %d", filename, res, buf_size);
    }
    result = (img *)malloc(sizeof(img));
    result->width = width;
    result->height = height;

    result->buffer = buffer;
    return result;
}

texture<pixel, 2, cudaReadModeElementType> tex;

KERNEL void bilinear_kernel(pixel* g_odata, int width, int height);
KERNEL void nop_kernel(pixel* g_result, int width, int height);

float cuda_bilinear(img *image) {
    Info("Performing CUDA bilinear interpolation");
    uint height = image->height;
    uint width = image->width;

    uint buf_size = width * height * RGB * sizeof(pixel);

    /* Setup raw image array */
    cudaChannelFormatDesc input_channel = cudaCreateChannelDesc<pixel>();
    cudaArray *d_raw_image = NULL;
    cutilSafeCall(cudaMallocArray(&d_raw_image, &input_channel, width * RGB, height));
    assert(d_raw_image);
    cutilSafeCall(cudaMemcpyToArray(d_raw_image, 0, 0,
                     image->buffer, buf_size, cudaMemcpyHostToDevice));

    /* Setup texture */
    cutilSafeCall(cudaBindTextureToArray(tex, d_raw_image));
    tex.normalized = 0;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;

    //CUresult cuTexRefSetAddressMode (CUtexref hTexRef, int Dim, CUaddress_mode am)
    //cuTexRefSetAddressMode(tex,0,CU_TR_ADDRESS_MODE_MIRROR);
    //cuTexRefSetAddressMode(tex,1,CU_TR_ADDRESS_MODE_MIRROR);
    //cuTexRefSetAddressMode(tex,2,CU_TR_ADDRESS_MODE_MIRROR);

    /* Setup output */
    pixel *d_result = NULL;
    cutilSafeCall(cudaMalloc((void **)&d_result, buf_size));

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
    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    //Info("Running test");
    bilinear_kernel<<< block_grid, thread_block>>>(d_result, width, height);
    cutilCheckMsg("Kernel execution failed");

    (cudaThreadSynchronize());
    cutilCheckError(cutStopTimer(timer));

    //printf("%d, %d, %f\n", image->width, image->height, cutGetTimerValue(timer));
    Info("Processing time: %f (ms)", cutGetTimerValue(timer));
    float time = (cutGetTimerValue(timer));
    Info("%.2f Mpixels/sec", (width * height / (time/ 1000.0f) ) / 1e6);
    cutilCheckError( cutDeleteTimer( timer));

    /* Copy result from device */
    cutilSafeCall(
            cudaMemcpy( image->buffer, d_result, buf_size, cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaFree(d_result));
    cutilSafeCall(cudaFreeArray(d_raw_image));

    cudaThreadExit();
    return time;
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

int main(int argc, char **argv) {
    if (argc <= 1){
        exit(0);
    }
    img* image = read_image_from_file(argv[1]);
    float t_tex = cuda_bilinear(image);
    Info("Texture time: %f",t_tex);
}

