/*
 * ahd.cpp
 *
 *  Created on: 22 Apr 2010
 *      Author: sjf
 */
#include <math.h>

#include "base.h"
#include "image.h"
#include "bayer.h"
#include "sys/times.h"
#include "util.h"
#include "colorspace.h"
#include "limits.h"

#define outside(a,b)
//#define outside(str,s) {if (s > 256 || s < 0) Warn(str,s);};


void host_ahd_horz_rb_interpolate_green
       (img *image, pixel* destpix, int x, int y, int filter_color) {
  /* Filter color is red or blue Interpolate green channel horizontally */
  /* Use existing green values */
  double sum = (get_pixel(image,x-1,y)[G] +
                get_pixel(image,x+1,y)[G])/2.0;

  /* And use existing red/blue values and apply filter 'h' */
  pixel left = get_pixel(image,x-2,y)[filter_color];
  pixel right = get_pixel(image,x+2,y)[filter_color];
  pixel center = get_pixel(image,x,  y)[filter_color];
  sum += (-(left/4.0) + (center/2.0) + -(right/4.0))/4.0;

  outside("Horz: Green sum is %f",sum);

  destpix[G] = clampc(sum);
}

void host_ahd_vert_rb_interpolate_green
       (img *image, pixel* destpix, int x, int y, int filter_color) {
  /* Filter color is red or blue Interpolate green channel horizontally */
  /* Use existing green values */
  double sum = (get_pixel(image,x,y-1)[G] +
                get_pixel(image,x,y+1)[G])/2.0;

  /* And use existing red/blue values and apply filter 'h' */
  sum += (-get_pixel(image,x,y-2)[filter_color]/4.0 +
           get_pixel(image,x,  y)[filter_color]/2.0 +
          -get_pixel(image,x,y+2)[filter_color]/4.0)/4.0;

  outside("Vert: Green sum is %f",sum);

  destpix[G] = clampc(sum);
}


void host_ahd_rb_interpolate_rb
        (img *image, img *dest, pixel *destpix, int x, int y, int filter_color) {
  /* Filter color is red or blue, interpolate missing red or blue channel */
  /* This function operates the same for horiz and vert interpolation */
  int dest_color = (filter_color == R) ? B : R;
  /* Get the difference between the Red/Blue and Green channels */
  double sum = (get_pixel(image,x-1,y-1)[dest_color] - get_pixel(dest,x-1,y-1)[G]) +
               (get_pixel(image,x-1,y+1)[dest_color] - get_pixel(dest,x-1,y+1)[G]) +
               (get_pixel(image,x+1,y-1)[dest_color] - get_pixel(dest,x+1,y-1)[G]) +
               (get_pixel(image,x+1,y+1)[dest_color] - get_pixel(dest,x+1,y+1)[G]);

  /* Apply low pass filter to the difference */
  sum /= 4.0;
  /* Use interpolated green value */
  uint g = destpix[G];
  assert(g >= 0 && g<= 255);
  sum += (uint)destpix[G];
  outside("R/B R/B sum is %f",sum);

  destpix[dest_color] = clampc(round(sum));
}


void host_ahd_g_interpolate_rb
         (img *image, img *dest, pixel *destpix, int x, int y, int dest_color) {
  /* Filter color is green */
  /* Interpolate Red and Blue channels */
  /* This function operates the same for horz and vert interpolation */
  double sum = 0;
  int bayer = settings->bayer_type;
  /* Red/Green rows */
    if ((dest_color == R && ((even(y) && (bayer == RGGB || bayer == GRBG))   ||
                             ( odd(y) && (bayer == BGGR || bayer == GBRG)))) ||
        (dest_color == B && (( odd(y) && (bayer == RGGB || bayer == GRBG))   ||
                             (even(y) && (bayer == BGGR || bayer == GBRG))))){
      /* Use left and right pixels */
      /* Get the difference between the Red/Blue and Green
       * channel (use only the sampled Green values */
        sum = (get_pixel(image,x-1,y)[dest_color] - get_pixel(dest,x-1,y)[G]) +
              (get_pixel(image,x+1,y)[dest_color] - get_pixel(dest,x+1,y)[G]);
    } else {
      /* Use top and bottom values */
        sum = (get_pixel(image,x,y-1)[dest_color] - get_pixel(dest,x,y-1)[G]) +
              (get_pixel(image,x,y+1)[dest_color] - get_pixel(dest,x,y+1)[G]);
    }
    /* Apply low pass filter */
    sum /= 2.0;
    /* Use sampled green value */
    sum += destpix[G];

    outside("R/B R/B sum is %f",sum);

    destpix[dest_color] = clampc(round(sum));
}

void to_cielab(pixel *src_buf, float *dest, uint width, uint height){
  float L,a,b;

  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
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

void build_homo_map(float **cie, uchar **dest, uint width, uint height, int ball_dist){
    /* Neighbour hood size */
    int neigh_size = ball_area(ball_dist);

  float lum_diff[HORZ_VERT][neigh_size];
    float chrom_diff[HORZ_VERT][neigh_size];

  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
        for (int d = 0; d < HORZ_VERT; d++) {
            float *pix = get_cie(cie[d],x,y,width);
            /* Get the lum and chrom differences for the pixel in the
             * neighbourhood.
             */
            int i = 0;
            for (int dy = -ball_dist; dy <= ball_dist; dy++){
                for (int dx = -ball_dist; dx <= ball_dist; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    if (!inside(x+dx, y+dy, width, height)) {
                        lum_diff[d][i] = 0xFF;
                        chrom_diff[d][i] = 0xFF;

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

void show(int *a, int n) {
  int i =0;
  while (i < n-1){
    printf("%d, ",a[i]);
    i++;
  }
  printf("%d\n",a[i]);
}

void showD(int *a, int n) {
  int i =0;
  while (i < n){
      if (a[i] != 0)
          printf("%d(%d), ",i,a[i]);
    i++;
  }
  printf("\n");
}

void radix_sort_uint(unsigned int *a, size_t size/*, int bits*/)
{
    uint *a_orig = a;
    uint tempa[9];
    uint *temp = tempa;

    int bits = 4;
    int cntsize = 1u << 4;
    int cntarray[cntsize];

    int max_mask = 0x1ff;
    /* Improve performance by adapting max_mask to the
     * maximum value in the array.
     * uint max = 0;
       for (int i = 0; i < size; i++) {
         if (a[i] > max) {
            max = a[i];
         }
       }
     *
     */

    uint rshift = 0;
    for(uint mask=~(UINT_MAX<<bits); mask & max_mask; mask <<= bits, rshift += bits) {
        bzero(cntarray, cntsize * sizeof(int));

        for (uint i = 0; i < size; i++) {
            uint key=(a[i] & mask) >> rshift;
            ++(cntarray[key]);
        }
        for(int i=1; i < cntsize; ++i){
            cntarray[i] += cntarray[i-1];
        }
        for(int i = size-1; i >= 0; i--) {
            uint key = (a[i] & mask) >> rshift;
            temp[ cntarray[key]-1 ] = a[i];
            --cntarray[key];
        }

        uint *swap = temp;
        temp = a;
        a = swap;
    }
    if (a != a_orig) {
        memcpy(a_orig, a, size*sizeof(unsigned int));
    }
}

int median2(int *src, int n) {
    for (int i = 0; i < n; i++) {
        int x = src[i];
        if (x < 0) {
            x = 255+x;
        } else {
            x += 255;
        }
        src[i] = x;
    }
    radix_sort_uint((unsigned int *)src,n);

    int med = src[n/2];
    if (med < 255) {
        return -(255-med);
    }
    return med-255;
}


#define DEPTH 511
int distribution[DEPTH];
int median(int *src, int n) {
    bzero(distribution,sizeof(distribution));
    for (int i = 0; i < n; i++) {
        int x = src[i];
        //assert(x <= 255);
        //assert(x >= -255);
        if (x < 0) {
            x = 255+x;
            //assert(x >= 0);
            //assert(x < 255);

        } else {
            /* x >= 0 */
            x += 255;
            //assert (x >= 255);
            //assert (x < 511);
        }
        distribution[x]++;
        /*if (i == n-1) {
            distribution[x] = n;
        }*/
    }

    //showD(distribution,DEPTH);

    int med = n/2;
    int sum = 0;
    for (int i = 0; i < DEPTH; i++) {
        sum += distribution[i];
        if (sum > med) {
            if (i < 255) {
                return -(255-i);
            }
            return i-255;
        }
    }
    assert(0);
}

#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { int temp=(a);(a)=(b);(b)=temp; }
int median4(int * p) {
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ;
    PIX_SORT(p[0], p[1]) ; PIX_SORT(p[3], p[4]) ; PIX_SORT(p[6], p[7]) ;
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ;
    PIX_SORT(p[0], p[3]) ; PIX_SORT(p[5], p[8]) ; PIX_SORT(p[4], p[7]) ;
    PIX_SORT(p[3], p[6]) ; PIX_SORT(p[1], p[4]) ; PIX_SORT(p[2], p[5]) ;
    PIX_SORT(p[4], p[7]) ; PIX_SORT(p[4], p[2]) ; PIX_SORT(p[6], p[4]) ;
    PIX_SORT(p[4], p[2]) ; return(p[4]) ;
}

int median_diff(pixel *buf, int x, int y, int width, int height, int chan1, int chan2) {
  int diffs[9];
  int i = 0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (inside(x+dx, y+dy, width, height)) {
        pixel *pix = get_pix(buf,x+dx,y+dy,width);
        diffs[i] = pix[chan1] - pix[chan2];
        //printf("%d-%d = %d\n", pix[chan1], pix[chan2], diffs[i]);
        i++;
      }
    }
  }
  while (i<9) {
    // at the edges copy the first value
    diffs[i++] = diffs[0];
  }

  //qsort(diffs,i,sizeof(int),intcmp);
  //int m = diffs[i/2];
  //printf("%d,%d (%d) median: %d\n",x,y,i,m);
  //return m;

  /*Alternative median methods */
  //int m = median2(diffs,i); /* radix sort */
  //int m = median(diffs,i);  /* histogram */
  int m = median4(diffs);  /* sorting network */
  return m;

  /* For testing that median works correctly */
  /*
    for (int j = 0; j < i; j++) {
        int v = diffs[j];
        if (v < 255) {
            v = -(255-v);
        } else {
            v = v-255;
        }
        diffs[j] = v;
    }
   */
    qsort(diffs,i,sizeof(int),intcmp);
  int msq = diffs[i/2];
  if (m != msq && (i&1)) {
      show(diffs,i);
      printf("%d,%d (%d) median: %d\n",x,y,i,m);
      printf("median(qs): %d\n",msq);

  } else {
      //printf("OK: median msq: %d\n",m);
  }
  return m;
}

void remove_artifacts(pixel *buf, pixel *result, uint width, uint height) {
  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      pixel *src = get_pix(buf,x,y,width);
      pixel *dest = get_pix(result,x,y,width);
      dest[R] = clampc(median_diff(buf,x,y,width,height,R,G) + src[G]);
      dest[B] = clampc(median_diff(buf,x,y,width,height,B,G) + src[G]);

      dest[G] = clampc(round((median_diff(buf,x,y,width,height,G,R) +
                              median_diff(buf,x,y,width,height,G,B) +
                              src[R] + src[B])/2.0));
    }
  }
}

void choose_interp_direction(pixel * result, pixel **interpolated, uchar **homo_map,
    uint width, uint height, settings_t *settings){

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

void ahd_interpolate(img *image, img *horz, img *vert) {
    uint width = image->width;
    uint height = image->height;
    //startTimer(interg);
    /* Interpolate green channel first */
    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
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
        }
    }
    //stopTimer(interg);
    //startTimer(interrb);
    /* Interpolate red/green channel */
    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
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
    }
    //stopTimer(interrb)
}

void host_ahd(img *image){
    Info("Performing AHD interpolation");

    uint width = image->width;
    uint height = image->height;

    size_t bufsize = height * width * RGB * sizeof(pixel);
    img *horz = new_image(height,width);
    img *vert = new_image(height,width);
    horz->buffer = mallocz<pixel>(bufsize);
    vert->buffer = mallocz<pixel>(bufsize);


    pixel *interpolated[] = { horz->buffer, vert->buffer };
    ahd_interpolate(image, horz, vert);

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
      to_cielab(interpolated[d],cie_interpolated[d],width,height);
    }

    size_t homo_bufsize = height * width * sizeof(uchar);
    uchar *homo_horz = mallocz<uchar>(homo_bufsize);
    uchar *homo_vert = mallocz<uchar>(homo_bufsize);
    uchar *homo_map[] = { homo_horz, homo_vert };

    //build_homo_map(cie_horz, cie_vert, homo_horz, homo_vert, width, height);
    build_homo_map(cie_interpolated, homo_map, width, height, settings->ball_distance);

    if (settings->save_temps) {
      int max_homo = ball_area(settings->ball_distance);
      save_map(homo_horz,width,height,max_homo,"img/homo_map_horz.ppm");
      save_map(homo_vert,width,height,max_homo,"img/homo_map_vert.ppm");
    }

    //stopTimer(homo);
    pixel *interpolated_img = mallocz<pixel>(bufsize);

    //startTimer(direction);
    choose_interp_direction(interpolated_img, interpolated, homo_map, width, height, settings);
    //stopTimer(direction);

    uchar *filtered = image->buffer;//mallocz<uchar>(bufsize);
    for (uint i = 0; i < settings->median_filter_iterations; i++) {
        Info("Removing artifacts");
      //startTimer(artefact1);
      remove_artifacts(interpolated_img,filtered,width,height);
      //    stopTimer(artefact1);

      uchar *swap = interpolated_img;
      interpolated_img = filtered; filtered = swap;
    }
    if (settings->median_filter_iterations %2) {
      free(filtered);
    } else {
      free(image->buffer);
    }
    image->buffer = interpolated_img;

    free(horz->buffer);
    free(vert->buffer);
    free(cie_horz);
    free(cie_vert);
    free(homo_horz);
    free(homo_vert);
}

