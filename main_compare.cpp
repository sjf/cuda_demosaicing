/*
 * main.c
 *
 *  Created on: 27 Jan 2010
 *      Author: sjf65
 */

#include <getopt.h>
#include <math.h>
#include <string.h>

#include "base.h"
#include "image.h"
#include "util.h"

#include "bayer.h"

#include "bilinear.h"

#include "ahd.h"
#include "ahd_mask.h"

#include "bilinear.cuh"

#include "ahd.cuh"


settings_t _settings;
settings_t *settings = &_settings;

#define DIST_3(p,q) sqrt( SQ(p[0]-q[0]) + \
                      SQ(p[1]-q[1]) + \
                      SQ(p[2]-q[2]) )

#define DIST_2(p,q)  sqrt(SQ(p[1]-q[1]) + \
                      SQ(p[2]-q[2]) )

void compare(img* image1, img *image2, char *name, char *type) {
  if (image1->width != image2->width ||
    image1->height != image2->height) {
    Warn("Images are different sizes %dx%d and %dx%d",
        image1->width, image1->height, image2->width, image2->height);
    return;
  }

  int w = image1->width, h = image1->height;
  int cie_bufsize = h * w * LAB * sizeof(float);
  float *ref_cie = mallocz<float>(cie_bufsize);
  float *img_cie = mallocz<float>(cie_bufsize);
  to_cielab(image1->buffer,ref_cie,w,h);
  to_cielab(image2->buffer,img_cie,w,h);
  img *zip = new_image(h,w,NULL);

  int n = 0;

  for (int y = 1; y < h-1; y++) {
    for (int x = 1; x < w-1; x++) {

      float *pix = get_cie(ref_cie,x,y,w);
      int minx = x, miny = y;
      double minval = 99999;

      for (int dx = -1;dx<=1;dx++) {
        for (int dy = -1;dy<=1;dy++) {
          if (!(dx == 0 && dy == 0)) {
            float *p = get_cie(ref_cie,x+dx,y+dy,w);
            double d = DIST_2(pix,p);
            if (d <= minval) {
              minval = d;
              minx = x+dx;
              miny = y+dy;
            }
          }
        }
      }
      float *p_ref = get_cie(ref_cie,minx,miny,w);
      float *p_img = get_cie(img_cie,minx,miny,w);
      double psi = DIST_2(p_ref,p_img);
      if (ABS(psi) > 2.3) {
        // Zippering here
        n++;
        pixel *t = get_pixel(zip,x,y);
        t[R] = t[G] = t[B] = 255;
      }
    }
  }
  float np = (n * 100) / (w*h);

  printf("%s, %s, %0.2f\n",name,type,np);
  write_image_to_file(zip->buffer,w,h,"img/zippering.ppm");
}

char name[1024];

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: ref_img img2\n");
        exit(1);
    }

    char *file1 = argv[1];
    char *file2 = argv[2];

    img *image1 = read_image_from_file(file1);
    img *image2 = read_image_from_file(file2);

    strcpy(name,file1);
    char * ext = strchr(name,'.');
    if (*ext == '.') {
      *ext = '\0';
    }
    const char *type;
    if (strstr(file2,"ahd.")){
      type = "AHD";
    } else if (strstr(file2,"bilin.")) {
      type = "bil";
    } else if (strstr(file2,"ahdmask.")) {
      type = "Mask";
    } else if (strstr(file2,"ahdmask2.")) {
      type = "Mask2";
    } else {
      Error("Unknown interpolation type");
      exit(1);
    }

    compare(image1,image2, name, (char *)type);
}
