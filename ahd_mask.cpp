/*
 * ahd_mask.cpp
 *
 *  Created on: 8 Apr 2010
 *      Author: sjf
 */
#include <stdlib.h>
#include <math.h>

#include "base.h"
#include "image.h"
#include "bayer.h"
#include "sys/times.h"
#include "util.h"
#include "colorspace.h"
#include "limits.h"
#include "image.h"
#include "bilinear.h"
#include "ahd.h"
#include "mask.h"
#include "ahd_mask.h"

void bayer_to_grayscale(pixel *bayer, pixel *result, uint width, uint height) {
    uint gwidth = width/4;
    uint gheight = height/4;
    for (uint y = 0; y < gheight; y++) {
        for (uint x = 0; x < gwidth; x++) {
            int px = x *4;
            int py = y *4;
            pixel g = (get_pix(bayer,px+1,py,width)[G]  +
                       get_pix(bayer,px,py+1,width)[G])/2;
            pixel r =  get_pix(bayer,px,py,width)[R];
            pixel b =  get_pix(bayer,px+1,py+1,width)[B];

            pixel *res = get_gray(result,x,y,gwidth);
            *res = TO_GRAY(r,g,b);
            DebugI(*res);
        }
    }
}

void detect_edges(img *grayscale, pixel *edges){
    for (uint y = 0; y < grayscale->height; y++) {
        for (uint x = 0; x < grayscale->width; x++) {
            /* Sobel edge filter */
            uint gx  = abs(    get_gray_pixel(grayscale,x+1,y-1) +
                           2 * get_gray_pixel(grayscale,x+1,y) +
                               get_gray_pixel(grayscale,x+1,y+1) +

                              -get_gray_pixel(grayscale,x-1,y-1) +
                           2 *-get_gray_pixel(grayscale,x-1,y) +
                              -get_gray_pixel(grayscale,x-1,y+1));

            uint gy  = abs(    get_gray_pixel(grayscale,x-1,y-1) +
                           2 * get_gray_pixel(grayscale,x,  y-1) +
                               get_gray_pixel(grayscale,x+1,y-1) +

                              -get_gray_pixel(grayscale,x-1,y+1) +
                           2 *-get_gray_pixel(grayscale,x,  y+1) +
                              -get_gray_pixel(grayscale,x+1,y+1));
            pixel *res = get_gray(edges,x,y,grayscale->width);
            *res = (gx + gy > settings->edge_threshold) ? 255 : 00;
            //*res = clampc(gx + gy); /* without thresholding */
        }
    }
}

void mask_ahd(img *image){
    Info("Performing AHD Mask interpolation");

    uint width = image->width;
    uint height = image->height;

    /* Save bayer image for use with AHD+mask */
    size_t bufsize = height * width * RGB * sizeof(pixel);
    pixel *src = mallocz<pixel>(bufsize);
    memcpy(src,image->buffer,bufsize);
    /* Do bilinear interpolation on the whole image */
    host_bilinear(image);

    /* Create mask */
    /* Convert interpolated image to grayscale */
    size_t gscale_bufsize = height * width * sizeof(pixel);
    pixel *gray_scale = mallocz<pixel>(gscale_bufsize);
    img *gray_img = new_image(height,width,gray_scale);

    img_to_grayscale(image,gray_scale);
    if (settings->save_temps){
        save_grayscale(gray_scale,width,height,"img/bilinear_grayscale.ppm");
    }
    /* Detect edges */
    img *edge_img = new_image(height,width,NULL);
    detect_edges(gray_img, edge_img->buffer);
    if (settings->save_temps){
        save_grayscale(edge_img->buffer,width,height,"img/edges.ppm");
    }
    /* Dilate edges */
    pixel *dilated = mallocz<pixel>(gscale_bufsize);
    for (uint i = 0; i < settings->dilations; i++) {
        dilate(edge_img, dilated);
        SWAP(dilated,edge_img->buffer);
    }
    dilated = edge_img->buffer;

    if (settings->save_temps){
        save_mask(dilated,width,height);
    }
    /* Swap bayer image back */
    pixel *bilinear = image->buffer;
    image->buffer = src;
    /* Apply AHD interpolation to the masked areas */
    apply_mask_ahd(image,dilated,bilinear);

    free(src);
    free_image(gray_img);
    free_image(edge_img);
    return;
}
