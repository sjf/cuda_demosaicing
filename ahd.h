/*
 * ahd.h
 *
 *  Created on: 22 Apr 2010
 *      Author: sjf
 */


void host_ahd(img *image);

int median_diff(pixel *buf, int x, int y, int width, int height, int chan1, int chan2);


void host_ahd_horz_rb_interpolate_green
       (img *image, pixel* destpix, int x, int y, int filter_color);
void host_ahd_vert_rb_interpolate_green
       (img *image, pixel* destpix, int x, int y, int filter_color);
void host_ahd_rb_interpolate_rb
        (img *image, img *dest, pixel *destpix, int x, int y, int filter_color);
void host_ahd_g_interpolate_rb
        (img *image, img *dest, pixel *destpix, int x, int y, int dest_color);
void to_cielab(pixel *src_buf, float *dest, uint width, uint height);
