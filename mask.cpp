/*
 * mask.cpp
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

void dilate(img *edges, pixel *dilated){
  int mask_size = 0; // for debugging/logging
    for (uint y = 0; y < edges->height; y++) {
        for (uint x = 0; x < edges->width; x++) {
          int neighs = 0;
          for (int dy = -1; dy <= 1 && neighs < 1; dy++) {
            for (int dx = -1; dx <=1 && neighs < 1; dx++) {
              neighs += get_gray_pixel(edges,x+dx,y+dy);
            }
          }
          pixel *res = get_gray(dilated,x,y,edges->width);
          *res = (neighs) ? 255 : 0;
          mask_size += (neighs) ? 1 : 0;
        }
    }
    float size = edges->width * edges->height;
    float m = (float)mask_size*100.0/size;
    Info("Mask: %02.2f%% of image", m);
    printf("%s, %02.2f\n",settings->image_name,m);
    settings->mask_size = m;
}

/**
 *
 * Here follows the AHD component functions, the same as in ahd.cpp except with an extra mask
 * parameter.
 *
 */

void to_cielab_mask(pixel *src_buf, float *dest, uint width, uint height, pixel *mask){
  float L,a,b;

  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      if (get_gray_val(mask,x,y,width)) {
        pixel *pix = get_pix(src_buf,x,y,width);
        //rgb_to_cielab(pix[R],pix[G],pix[B],&L,&a,&b);
        cvRGBtoLab(pix[R],pix[G],pix[B],&L,&a,&b);

        float *destpix = get_cie(dest,x,y,width);
        destpix[Ls] = L;
        destpix[As] = a;
        destpix[Bs] = b;
      }
    }
  }
}

void build_homo_map_mask(float **cie, uchar **dest, uint width, uint height, int ball_dist, pixel *mask){
    /* Neighbour hood size */
    int neigh_size = ball_area(ball_dist);

  float lum_diff[HORZ_VERT][neigh_size];
    float chrom_diff[HORZ_VERT][neigh_size];

  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      if (get_gray_val(mask,x,y,width)) {

        for (int d = 0; d < HORZ_VERT; d++) {

          float *pix = get_cie(cie[d],x,y,width);
          /* Get the lum and chrom differences for the pixel in the
           * neighbourhood.
           */
          int i = 0;
          for (int dy = -ball_dist; dy <= ball_dist; dy++){
            for (int dx = -ball_dist; dx <= ball_dist; dx++) {
              if (dx == 0 && dy == 0) continue;

              if (!get_gray_val(mask,x,y,width) ||
                !inside(x+dx, y+dy, width, height)) {
                lum_diff[d][i] = 0;
                chrom_diff[d][i] = 0;

              } else {
                //printf("i: %d dx,dy: %d,%d\n",i,dx,dy);
                float *neigh = get_cie(cie[d],x+dx,y+dy,width);

                lum_diff[d][i] = diff1(pix[Ls], neigh[Ls]);
                chrom_diff[d][i] = diff2(pix[As],pix[Bs],
                    neigh[As],neigh[Bs]);
              }
              i++;
            }
          }
        }
        /* Homogenity differences have been calculated for horz and vert directions */
        /* Find the adaptive thresholds, the same threshold is used for horz and vert */
        /* Horizontal case, look at left and right values */
        /* Vertical case, look at top, bottom values */
        int h1,h2,v1,v2;
        if (ball_dist == 1) {
          h1 = 3; h2 = 4;
          v1 = 1; v2 = 6;
        } else {
          /* d == 2 */
          h1 = 11; h2 = 12;
          v1 = 7; v2 = 16;
        }
        float lum_thres = MIN(MAX(lum_diff[HORZ][h1],lum_diff[HORZ][h2]),
                          MAX(lum_diff[VERT][v1],lum_diff[VERT][v2]));

        float chrom_thres = MIN(MAX(chrom_diff[HORZ][h1],chrom_diff[HORZ][h2]),
                            MAX(chrom_diff[VERT][v1],chrom_diff[VERT][v2]));

        /* Calculate the number of pixels in the homogenity neighbourhood
         * (number below the threshold). */
        /* This is the homogenity value for this pixel. */
        for (int d = 0; d < HORZ_VERT; d++) {
          int n_homo = 0;
          for (int i = 0; i < neigh_size; i++) {
            if (lum_diff[d][i] <= lum_thres &&
                chrom_diff[d][i] <= chrom_thres){
              ++n_homo;
            }
          }
          *(get_homo(dest[d],x,y,width)) = n_homo;
        }
      }
    }
  }
}

void choose_interp_direction_mask(pixel * result, pixel **interpolated, uchar **homo_map,
    uint width, uint height, settings_t *settings, pixel *mask){

  int score_bufsize = width * height * sizeof(uchar);

  /* These are just for debugging */
  uchar *horz_score_map = NULL;
  uchar *vert_score_map = NULL;
  uchar *direction_map  = NULL;
  if (settings->save_temps) {
    horz_score_map = mallocz<uchar>(score_bufsize);
    vert_score_map = mallocz<uchar>(score_bufsize);
    direction_map  = mallocz<uchar>(score_bufsize);
  }
    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
          if (get_gray_val(mask,x,y,width)) {
          int horz_score = 0;
          int vert_score = 0;
      for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++) {
          if (inside(x+dx,y+dy,width,height)){
            // todo divide the score by the area so that this
            // works properly at the borders
            horz_score += *(get_homo(homo_map[HORZ],x+dx,y+dy,width));
            vert_score += *(get_homo(homo_map[VERT],x+dx,y+dy,width));
          }
        }
      }
      if (settings->save_temps) {
        //printf("Horz score: %d, vert score: %d\n", horz_score, vert_score);
        *get_homo(horz_score_map,x,y,width) = horz_score;
        *get_homo(vert_score_map,x,y,width) = vert_score;
      }
      pixel *h_src = get_pix(interpolated[HORZ],x,y,width);
      pixel *v_src = get_pix(interpolated[VERT],x,y,width);
      pixel *dest = get_pix(result,x,y,width);
      /*if (horz_score == vert_score) {
          Info("Averaging homo vals %d,%d",x,y);
        dest[R] = round(h_src[R]/2.0 + v_src[R]/2.0);
        dest[G] = round(h_src[G]/2.0 + v_src[G]/2.0);
        dest[B] = round(h_src[B]/2.0 + v_src[B]/2.0);
      } else { */

      int dir = HORZ;
      pixel *src = h_src;
      if (vert_score > horz_score) {
        src = v_src;
        dir = VERT;
      }
      if (settings->save_temps) {
        uchar *d_dest = get_homo(direction_map,x,y,width);
        *d_dest = dir;
      }

      memcpy(dest,src,RGB*sizeof(pixel));
      /*}*/
        }
        }
    }
    if (settings->save_temps) {
      save_map(horz_score_map,width,height,63,"img/score_horz.ppm");
      save_map(vert_score_map,width,height,63,"img/score_vert.ppm");
      save_map(direction_map,width,height,1,"img/direction.ppm");
      write_image_to_file(result,width,height,"img/pre_noise.ppm");

      free(horz_score_map);
      free(vert_score_map);
      free(direction_map);
    }
}

void remove_artifacts_mask(pixel *buf, pixel *result, uint width, uint height, pixel *mask) {
  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      pixel *src = get_pix(buf,x,y,width);
      pixel *dest = get_pix(result,x,y,width);
      if (get_gray_val(mask,x,y,width)) {
        dest[R] = clampc(median_diff(buf,x,y,width,height,R,G) + src[G]);
        dest[B] = clampc(median_diff(buf,x,y,width,height,B,G) + src[G]);
        dest[G] = clampc(round((median_diff(buf,x,y,width,height,G,R) +
                                median_diff(buf,x,y,width,height,G,B) +
                                src[R] + src[B])/2.0));
      } else {
        // Just copy pixel from source
        memcpy(dest,src,sizeof(pixel)*RGB);
      }
    }
  }
}

void apply_mask_ahd(img *image, pixel *mask, pixel *bilinear_result) {
  Info("Performing AHD interpolation (with mask)");
//  struct tms start, end;
//  clock_t starttime, endtime;
//  starttime = times(&start);

  uint width = image->width;
  uint height = image->height;

  size_t bufsize = height * width * RGB * sizeof(pixel);
  img *horz = new_image(height,width);
  img *vert = new_image(height,width);
  horz->buffer = mallocz<pixel>(bufsize);
  vert->buffer = mallocz<pixel>(bufsize);


  //startTimer(interg);
  pixel *interpolated[] = { horz->buffer, vert->buffer };

  /* Interpolate green channel first */
  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      //printf("%d,%d mask: .%d.\n",x,y,get_gray_val(mask,x,y,width));


      //if (get_gray_val(mask,x,y,width)) {
        pixel *pix = get_pixel(image,x,y);
        pixel *horz_dest = get_pixel(horz,x,y);
        pixel *vert_dest = get_pixel(vert,x,y);

        int filter_color = get_filter_color(x,y);
        //if (filter_color == G){
        /* Copy existing values */
        horz_dest[filter_color] = pix[filter_color];
        vert_dest[filter_color] = pix[filter_color];
        //}
        /* Red/Blue only pixels, interpolate green */
        if (filter_color != G) {
          /* Horz */
          host_ahd_horz_rb_interpolate_green(image,horz_dest,x,y,filter_color);
          /* Vert */
          host_ahd_vert_rb_interpolate_green(image,vert_dest,x,y,filter_color);
        }
      //}
    }
  }
  //stopTimer(interg);
  //startTimer(interrb);
  /* Interpolate red/green channel */
  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      //if (get_gray_val(mask,x,y,width)) {
        int filter_color = get_filter_color(x,y);

        pixel *horz_dest = get_pixel(horz,x,y);
        pixel *vert_dest = get_pixel(vert,x,y);

        //assert(pix[filter_color] == 255);

        /* Red/Blue only pixels, interpolate red or blue
         * (Green has already been interpolated) */
        if (filter_color == R || filter_color == B) {

          /* Horz */
          host_ahd_rb_interpolate_rb(image,horz,horz_dest,x,y,filter_color);
          /* Vert */
          host_ahd_rb_interpolate_rb(image,vert,vert_dest,x,y,filter_color);
        } else {
          /* Green only pixels, interpolate red and blue */
          /* Horz */
          host_ahd_g_interpolate_rb(image,horz,horz_dest,x,y,R);
          host_ahd_g_interpolate_rb(image,horz,horz_dest,x,y,B);
          /* Vert */
          host_ahd_g_interpolate_rb(image,vert,vert_dest,x,y,R);
          host_ahd_g_interpolate_rb(image,vert,vert_dest,x,y,B);
        }
      }
    //}
  }


  //stopTimer(interrb)
  if (settings->save_temps) {
    write_image_to_file(horz->buffer,width,height,"img/interpolation_horz.ppm");
    write_image_to_file(vert->buffer,width,height,"img/interpolation_vert.ppm");
  }

  size_t cie_bufsize = height * width * LAB * sizeof(float);
  float *cie_horz = mallocz<float>(cie_bufsize);
  float *cie_vert = mallocz<float>(cie_bufsize);
  float *cie_interpolated[] = { cie_horz, cie_vert };

  //startTimer(homo);
  for (int d = 0; d < HORZ_VERT; d++) {
    to_cielab_mask(interpolated[d],cie_interpolated[d],width,height,mask);
  }

  size_t homo_bufsize = height * width * sizeof(uchar);
  uchar *homo_horz = mallocz<uchar>(homo_bufsize);
  uchar *homo_vert = mallocz<uchar>(homo_bufsize);
  uchar *homo_map[] = { homo_horz, homo_vert };

  //build_homo_map(cie_horz, cie_vert, homo_horz, homo_vert, width, height);
  build_homo_map_mask(cie_interpolated, homo_map, width, height, settings->ball_distance,mask);

  if (settings->save_temps) {
    int max_homo = ball_area(settings->ball_distance);
    save_map(homo_horz,width,height,max_homo,"img/homo_map_horz.ppm");
    save_map(homo_vert,width,height,max_homo,"img/homo_map_vert.ppm");
  }

  //stopTimer(homo);

  //startTimer(direction);
  choose_interp_direction_mask(bilinear_result, interpolated, homo_map, width, height, settings,mask);
  //stopTimer(direction);

  pixel *denoised = mallocz<pixel>(bufsize);

  for (uint i = 0; i < settings->median_filter_iterations; i++) {
    Info("Removing artifacts");
    //startTimer(artefact1);
    remove_artifacts_mask(bilinear_result,denoised,width,height,mask);
    //    stopTimer(artefact1);
    SWAP(denoised,bilinear_result);
  }


  free(horz->buffer);
  free(vert->buffer);
  free(cie_horz);
  free(cie_vert);
  free(homo_horz);
  free(homo_vert);

  image->buffer = denoised;

//  endtime = times(&end);
//  show_times(image, &start,&end,endtime-starttime, settings);
}
