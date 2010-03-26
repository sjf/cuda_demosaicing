/*
 * colorspace.h
 *
 *  Created on: 11 Mar 2010
 *      Author: sjf
 */

#ifndef COLORSPACE_H_
#define COLORSPACE_H_
void rgb_to_cielab(unsigned char r, unsigned char g, unsigned char b,
                   float *L, float *a, float *b_);

void cielab_to_rgb(float L, float a, float b_,
                   unsigned char *r, unsigned char *g, unsigned char *b);

void cvRGBtoLab(unsigned char r1, unsigned char g1, unsigned char b1,
                float *L_, float *a_, float *b_);

#endif /* COLORSPACE_H_ */
