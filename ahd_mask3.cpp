/*
 * ahd_mask2.cpp
 *
 *  Created on: 14 May 2010
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

//G = (11*R + 16*G + 5*B) /32
#define R_LUM 11
#define G_LUM 16
#define B_LUM 5
#define LUM_SUM 32

int dist3_(int a, int b, int c, int a1, int b1, int c1){
    return abs(SQ(a-a1) + SQ(b-b1) + SQ(c-c1));
}

void detect_edges_w_laplacian(img *bilin, pixel *edges){
    for (uint y = 0; y < bilin->height; y++) {
        for (uint x = 0; x < bilin->width; x++) {
            /* Sobel edge filter */
            uint gx  = abs(    get_pixel(bilin,x+1,y-1)[G] +
                           2 * get_pixel(bilin,x+1,y)[G] +
                               get_pixel(bilin,x+1,y+1)[G] +

                              -get_pixel(bilin,x-1,y-1)[G] +
                           2 *-get_pixel(bilin,x-1,y)[G] +
                              -get_pixel(bilin,x-1,y+1)[G]);

            uint gy  = abs(    get_pixel(bilin,x-1,y-1)[G] +
                           2 * get_pixel(bilin,x,  y-1)[G] +
                               get_pixel(bilin,x+1,y-1)[G] +

                              -get_pixel(bilin,x-1,y+1)[G] +
                           2 *-get_pixel(bilin,x,  y+1)[G] +
                              -get_pixel(bilin,x+1,y+1)[G]);
            pixel *res = get_gray(edges,x,y,bilin->width);
            *res = (gx + gy > settings->edge_threshold) ? 255 : 00;
            *res = clampc(gx + gy); /* without thresholding */


//            /* Laplacian edge filter */
//            /* Detects edges, but not direction */
////            uint g  = -(get_gray_pixel(grayscale,x+1,y-1) +
////                        2*get_gray_pixel(grayscale,x+1,y) +
////                        get_gray_pixel(grayscale,x+1,y+1) +
////
////                       2*get_gray_pixel(grayscale,x,y-1) +
////                       2*get_gray_pixel(grayscale,x,y+1) +
////
////                        get_gray_pixel(grayscale,x-1,y-1) +
////                       2*get_gray_pixel(grayscale,x-1,y) +
////                       get_gray_pixel(grayscale,x-1,y+1));
//
////                       get_gray_pixel(grayscale,x+1,y) +
////                       get_gray_pixel(grayscale,x-2,y) +
////                       get_gray_pixel(grayscale,x,y+2) +
////                       get_gray_pixel(grayscale,x,y-2));
////            g += 12 * get_gray_pixel(grayscale,x,y);
////
////            g+=255;
//
//            /* Laplacian edge filter */
//            /* Detects edges, but not direction */
//            int gx  = -(get_gray_pixel(grayscale,x-1,y-1) +
//                        get_gray_pixel(grayscale,x-1,y) +
//                        get_gray_pixel(grayscale,x-1,y+1));
//
//            gx +=     (get_gray_pixel(grayscale,x,y-1) +
//                       get_gray_pixel(grayscale,x,y+1) +
//
//                      get_gray_pixel(grayscale,x+1,y-1) +
//                      get_gray_pixel(grayscale,x+1,y) +
//                      get_gray_pixel(grayscale,x+1,y+1));
//
//            gx -= 2 * get_gray_pixel(grayscale,x,y);
//
//
//            int gy  = -(get_gray_pixel(grayscale,x-1,y-1) +
//                        get_gray_pixel(grayscale,x  ,y-1) +
//                        get_gray_pixel(grayscale,x+1,y-1));
//
//            gy +=     (get_gray_pixel(grayscale,x+1,y) +
//                       get_gray_pixel(grayscale,x-1,y) +
//
//                      get_gray_pixel(grayscale,x+1,y+1) +
//                      get_gray_pixel(grayscale,x  ,y+1) +
//                      get_gray_pixel(grayscale,x-1,y+1));
//
//            gy -= 2 * get_gray_pixel(grayscale,x,y);
//            //g += 255;
//            int g = MAX(gx,gy);
//            pixel *res = get_gray(edges,x,y,grayscale->width);
//            /*res = (g > settings->edge_threshold) ? 255 : 00;
//            *res = clampc(g); /* without thresholding */
        }
    }
}


void mask_ahd3(img *image){
    Info("Performing AHD Mask v3 interpolation");

    uint width = image->width;
    uint height = image->height;

    /* Save bayer image for use with AHD+mask */
    size_t bufsize = height * width * RGB * sizeof(pixel);
    pixel *src = mallocz<pixel>(bufsize);
    memcpy(src,image->buffer,bufsize);

    /* Do bilinear interpolation on the whole image */
    host_bilinear(image);

    /** Create mask **/
    /* Convert interpolated image to grayscale */
    size_t gscale_bufsize = height * width * sizeof(pixel);
//    pixel *gray_scale = mallocz<pixel>(gscale_bufsize);
//    img *gray_img = new_image(height,width,gray_scale);

//    img_to_grayscale(image,gray_scale);
//    if (settings->save_temps){
//        save_grayscale(gray_scale,width,height,"img/bilinear_grayscale.ppm");
//    }
    /* Detect edges */
    img *edge_img = new_image(height,width,NULL);
    detect_edges_w_laplacian(image, edge_img->buffer);
    if (settings->save_temps){
        save_grayscale(edge_img->buffer,width,height,"img/edges_v3.ppm");
    }
    /* Dilate edges */

    pixel *dilated = mallocz<pixel>(gscale_bufsize);
    for (uint i = 0; i < settings->dilations; i++) {
        dilate(edge_img, dilated);
        SWAP(dilated,edge_img->buffer);
    }
    dilated = edge_img->buffer;

    if (settings->save_temps){
        save_grayscale(dilated,width,height,"img/mask_v3.ppm");
    }
    /* Swap bayer image back */
    pixel *bilinear = image->buffer;
    image->buffer = src;
    /* Apply AHD interpolation to the masked areas */
    apply_mask_ahd(image,dilated,bilinear);

    free(src);
    free_image(edge_img);
    return;
}
