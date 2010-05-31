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

int dist3(int a, int b, int c, int a1, int b1, int c1){
    return abs(SQ(a-a1) + SQ(b-b1) + SQ(c-c1));
}


void detect_edges_w_contrast(img *bilinear, pixel *edges){
    for (uint y = 0; y < bilinear->height; y++) {
        for (uint x = 0; x < bilinear->width; x++) {
            /* Look at the 3x3 neighbours */
            /* Get the distance */
            int dist = 0;
            pixel *p = get_pixel(bilinear,x,y);
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (!(dx == 0 && dy == 0)) {
                        pixel *p1 = get_pixel(bilinear,x+dx,y+dy);
                        dist += dist3(p[R], p[G], p[B],
                                      p1[R],p1[G],p1[B]);
                    }
                }
            }
            dist /= 9;
            pixel *res = get_gray(edges,x,y,bilinear->width);
            *res = (dist > settings->edge_threshold) ? 255 : 00;
            //*res = clampc(d); /* without thresholding */
        }
    }
}

void mask_ahd2(img *image){
    Info("Performing AHD Mask v2 interpolation");

    uint width = image->width;
    uint height = image->height;

    /* Save bayer image for use with AHD+mask */
    size_t bufsize = height * width * RGB * sizeof(pixel);
    pixel *src = mallocz<pixel>(bufsize);
    memcpy(src,image->buffer,bufsize);

    /* Do bilinear interpolation on the whole image */
    host_bilinear(image);

    /** Create mask **/
    /* Detect edges */
    img *edge_img = new_image(height,width,NULL);
    detect_edges_w_contrast(image, edge_img->buffer);
    if (settings->save_temps){
        save_grayscale(edge_img->buffer,width,height,"img/edges_v2.ppm");
    }
    /* Dilate edges */
    size_t gscale_bufsize = height * width * sizeof(pixel);
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
    free_image(edge_img);
    return;
}
