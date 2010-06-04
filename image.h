/*
 * image.h
 *
 *  Created on: 28 Jan 2010
 *      Author: sjf65
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include "base.h"

#define R 0
#define G 1
#define B 2
#define RGB 3

#define Ls 0
#define As 1
#define Bs 2
#define LAB 3

#define HORZ 0
#define VERT 1
#define HORZ_VERT 2

#define mirror(a,max) ((a) < 0) ? -(a) : (((a) >= (max)) ? (max)-((a)%(max))-2 : (a))

//#define clamp(a,min,max) (a < min) ? min : ((a > max) ? max : a)
#define clampc(a) ((a) < 0) ? 0 : (((a) > 255) ? 255 : (uchar)(a))

#define inside(x,y,width,height) ((x)>=0 && (y)>=0 && (x)<(width) && (y)<(height))

//typedef unsigned int pix;
//typedef pix pixel[RGB];

typedef uchar pixel;

typedef struct {
    uint width;
    uint height;
    pixel *buffer;
} img;

inline uchar *get_pixel(img *image, int x, int y) {
    y = mirror(y,(int)image->height);
    x = mirror(x,(int)image->width);
    return image->buffer + (y * image->width + x)*RGB;
}

inline pixel get_gray_pixel(img *image, int x, int y) {
    y = mirror(y,(int)image->height);
    x = mirror(x,(int)image->width);
    return *(image->buffer + (y * image->width + x));
}


#define get_pix(buffer,x,y,width) ((buffer) + ((y) * (width) + (x))*RGB)
//inline uchar *get_pix(pixel *buffer, int x, int y, uint width) {
//  return buffer + (y * width + x)*RGB;
//}

#define get_cie(buffer,x,y,width) ((buffer) + ((y) * (width) + (x))*LAB)
//static inline float *get_cie(float *buffer, int x, int y, uint width) {
//  return buffer + (y * width + x)*LAB;
//}
#define get_array(buffer,x,y,width) ((buffer) + ((y) * (width) + (x)))

#define get_homo(buffer,x,y,width) ((buffer) + ((y) * (width) + (x)))
//static inline uchar *get_homo(uchar *buffer, int x, int y, uint width) {
//  return buffer + (y * width + x);
//}
//static inline float *get_float(float *buffer, int x, int y, uint width) {
//  return buffer + (y * width + x);
//}

#define get_gray(buffer,x,y,width) ((buffer) + ((y) * (width) + (x)))

#define get_gray_val(buffer,x,y,width) (*((buffer) + ((y) * (width) + (x))))

#define diff1(a,b) ((a) > (b)) ? ((a)-(b)) : ((b)-(a))
//static inline float diff1(float a, float b) {
//  float diff = a - b;
//  if (diff >= 0) return diff;
//  return -diff;
//}

#define diff2(a1, a2, b1, b2) (((a1) - (b1)) * ((a1) - (b1)) + ((a2) - (b2)) * ((a2) - (b2)));
//static inline float diff2(float a1, float a2, float b1, float b2) {
//  return (a1 - b1) * (a1 - b1) +
//       (a2 - b2) * (a2 - b2);
//}

// ball area is: (2*r + 1)^2 - 1
#define ball_area(r)  (4 * ((r)*(r) + (r)))

pixel *get_pixel(img *image, int x, int y);

img *new_image(uint height, uint width);
img *new_image(uint height, uint width, pixel *buffer);

img *read_image_from_file(char *filename);

int write_result_to_file(const char *filename, img *image);

int write_image_to_file(pixel *buf, uint width, uint height, const char *filename);

void save_map(uchar *buff, uint width, uint height, float max_val, const char *filename);

void save_grayscale(uchar *buff, uint width, uint height, const char *filename);

void free_image(img *image);

pixel *pad_image(pixel *src, uint width, uint height, int border);

void save_mask(uchar *buff, uint width, uint height);
#define R_LUM 11
#define G_LUM 16
#define B_LUM 5
#define LUM_SUM 32
#define TO_GRAY(r,g,b) (((r)*R_LUM + (g)*G_LUM + (b)*B_LUM)/LUM_SUM)
void img_to_grayscale(img *image, pixel *grayscale);
#endif /* IMAGE_H_ */
