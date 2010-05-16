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
#include "ahd_mask2.h"
#include "ahd_mask4.h"
#include "ahd_mask.h"

//G = (11*R + 16*G + 5*B) /32
#define R_LUM 11
#define G_LUM 16
#define B_LUM 5
#define LUM_SUM 32

void detect_edges_w_contrast4(img *bilinear, img* blurred, pixel *result){
    for (uint y = 0; y < bilinear->height; y++) {
        for (uint x = 0; x < bilinear->width; x++) {
            /* Look at the 3x3 neighbours */
            /* Get the distance */
            float dist = 0;
            float check_dist = 0;
            pixel *p =  get_pixel(bilinear,x,y);
            pixel *p2 = get_pixel(blurred,x,y);
            int g1 = TO_GRAY(p[R],p[G],p[B]);
            int g2 = TO_GRAY(p2[R],p2[G],p2[B]);

            pixel *res = get_gray(result,x,y,bilinear->width);

            if (abs(g1-g2) > settings->edge_threshold) {
                *res = 255;
            } else {
                *res  = 0;
            }
        }
    }
}

void scale_bayer(img *image, img* result) {
    int h = image->height;
    int w = image->width;
    int h2 = result->height;
    int w2 = result->width;
    for (uint y = 0; y < h2; y++) {
        for (uint x = 0; x < w2; x++) {
            pixel filter_color = get_filter_color(x,y);
            int val = 0;
            int xi = x*2, yi = y*2;
            if (filter_color == G) {
                if (odd(y)) {
                    yi -= 1;
                } else {
                    /* odd x */
                    xi -= 1;
                }
                /* center val */
                val = get_pixel(image,xi,yi)[G] * 2;
                /* diagonals */
                val += get_pixel(image,xi-1,yi-1)[G];
                val += get_pixel(image,xi+1,yi-1)[G];
                val += get_pixel(image,xi-1,yi+1)[G];
                val += get_pixel(image,xi+1,yi+1)[G];
                val /= 6;
            } else if (filter_color == R) {
                /* center val */
                val = get_pixel(image,xi,yi)[R] * 4;
                /* left/right/top/bottom neighbours */
                val += get_pixel(image,xi-2,yi)[R];
                val += get_pixel(image,xi+2,yi)[R];
                val += get_pixel(image,xi,yi-2)[R];
                val += get_pixel(image,xi,yi+2)[R];
                val /= 8;
            } else {
                xi -= 1;
                yi -= 1;
                /* center val */
                val = get_pixel(image,xi,yi)[B] * 4;
                /* left/right neighbours */
                val += get_pixel(image,xi-2,yi)[B];
                val += get_pixel(image,xi+2,yi)[B];
                val += get_pixel(image,xi,yi-2)[B];
                val += get_pixel(image,xi,yi+2)[B];
                val /= 8;
            }
            pixel *res = get_pixel(result,x,y);
            res[filter_color] = clampc(val);
        }
    }
}

void blur_bayer(img *image, img* result) {
    int h2 = result->height;
    int w2 = result->width;
    for (uint y = 0; y < h2; y++) {
        for (uint x = 0; x < w2; x++) {
            pixel filter_color = get_filter_color(x,y);
            int val = 0, n = 0;
            for (int dx = -3;dx<=3;dx++) {
                for (int dy = -3;dy<=3;dy++) {
                    int xi = x+dx;
                    int yi = y+dy;
                    if (filter_color == get_filter_color(xi,yi)) {
                        val += get_pixel(image,xi,yi)[filter_color];
                        n++;
                    }
                }
            }
            val /= n;
            pixel *res = get_pixel(result,x,y);
            res[filter_color] = clampc(val);
        }
    }
}

void scale_up(img *image, img* result) {
    int h = image->height;
    int w = image->width;
    int h2 = result->height;
    int w2 = result->width;
    for (uint y = 0; y < h2; y++) {
        for (uint x = 0; x < w2; x++) {
            for (int c = 0; c < RGB; c++) {
                int val = 0;
                for (int dx = -1;dx<=1;dx++) {
                    for (int dy = -1;dy<=1;dy++) {
                        int xi = round((float)x/2.0) + dx;
                        int yi = round((float)y/2.0) + dy;
                        val += get_pixel(image,xi,yi)[c] ;
                    }
                }
                val /= 9;
                pixel *res = get_pixel(result,x,y);
                res[c] = clampc(val);
            }
        }
    }
}

void blur(img *image, img*result) {
    int h = image->height;
    int w = image->width;
    for (uint y = 0; y < h; y++) {
        for (uint x = 0; x < w; x++) {
            for (uint c = 0; c < RGB; c++){
                int val = 0;
                for (int dx = -1;dx<=1;dx++) {
                    for (int dy = -1;dy<=1;dy++) {
                        val += get_pixel(image,x,y)[c];
                    }
                }
                val /= 9;
                pixel *res = get_pixel(result,x,y) + c;
                *res = clampc(val);
            }
        }
    }
}

void mask_ahd3b(img *image){
    Info("Performing AHD Mask v4 interpolation");

    uint width = image->width;
    uint height = image->height;

    /* Save bayer image for use with AHD+mask */
    size_t bufsize = height * width * RGB * sizeof(pixel);
    pixel *src = mallocz<pixel>(bufsize);
    memcpy(src,image->buffer,bufsize);

    /* Do bilinear interpolation on the whole image */
    host_bilinear(image);
    int w = image->width, h = image->height;
    int w2 = w/2;
    int h2 = h/2;

    img *bbayer = new_image(h,w,NULL);
    blur_bayer(image,bbayer);
    if (settings->save_temps) {
        write_image_to_file(bbayer->buffer,w,h,"img/bayer_blurred.ppm");
    }
    host_bilinear(bbayer);
    if (settings->save_temps) {
        write_image_to_file(bbayer->buffer,w,h,"img/bilin_blurred.ppm");
    }


    /** Create mask **/
    /* Detect edges */
    img *edge_img = new_image(height,width,NULL);
    detect_edges_w_contrast4(image,bbayer, edge_img->buffer);
    if (settings->save_temps){
        save_grayscale(edge_img->buffer,width,height,"img/edges_v4.ppm");
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
        save_grayscale(dilated,width,height,"img/mask_v4.ppm");
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
