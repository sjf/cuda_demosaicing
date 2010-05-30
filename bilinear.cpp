/*
 * bilinear.cpp
 *
 *  Created on: 22 Apr 2010
 *      Author: sjf
 */

#include "base.h"
#include "image.h"
#include "bayer.h"
#include "sys/times.h"
#include "util.h"

void host_bilinear(img *image){
    Info("Performing bilinear interpolation");
    for (uint y = 0; y < image->height; y++) {
        for (uint x = 0; x < image->width; x++) {

            pixel *pix = get_pixel(image,x,y);
            int filter_color = get_filter_color(x,y);

            double sum = 0;


            if (filter_color == R || filter_color == B) {
                /* Red/Blue only pixels */
                /* Green channel */
                sum = get_pixel(image,x-1,y)[G] +
                      get_pixel(image,x+1,y)[G] +
                      get_pixel(image,x,y-1)[G] +
                      get_pixel(image,x,y+1)[G];
                pix[G] = clampc(sum / 4);

                int dest_color = (filter_color == R) ? B : R;
                /* Red/Blue channel */
                sum = get_pixel(image,x-1,y-1)[dest_color] +
                      get_pixel(image,x-1,y+1)[dest_color] +
                      get_pixel(image,x+1,y-1)[dest_color] +
                      get_pixel(image,x+1,y+1)[dest_color];

                pix[dest_color] = clampc(sum / 4);
            } else {
                /* Green only pixels */
                /* Red/Green rows */
                int bayer = settings->bayer_type;
                if ((even(y) && (bayer == RGGB || bayer == GRBG)) ||
                    ( odd(y) && (bayer == GBRG || bayer == BGGR))) {
                    sum = get_pixel(image,x-1,y)[R] +
                          get_pixel(image,x+1,y)[R];
                    pix[R] = clampc(sum / 2);

                    sum = get_pixel(image,x,y-1)[B] +
                          get_pixel(image,x,y+1)[B];
                    pix[B] = clampc(sum / 2);
                } else {
                    sum = get_pixel(image,x,y-1)[R] +
                          get_pixel(image,x,y+1)[R];
                    pix[R] = clampc(sum / 2);

                    sum = get_pixel(image,x-1,y)[B] +
                          get_pixel(image,x+1,y)[B];
                    pix[B] = clampc(sum / 2);

                }
            }
        }
    }
}
