/*
 * ahd_kernels.cu
 *
 *  Created on: 26 Mar 2010
 *      Author: sjf
 */

#include "ahd.cuh"
#include "colorspace.cu"
#include "image.h"
#include "limits.h"

texture<pixel, 2, cudaReadModeElementType> src;
texture<pixel4, 2, cudaReadModeElementType> src_g;
texture<float4, 2, cudaReadModeElementType> horz_tex;
texture<float4, 2, cudaReadModeElementType> vert_tex;
texture<uchar, 2, cudaReadModeElementType> homo_h_tex;
texture<uchar, 2, cudaReadModeElementType> homo_v_tex;



//#define tex_get_comp(tex,x,y,c) tex2D((src),(mirror((x),width))*3+(c),(mirror((y),height)))
//#define tex_get_comp(tex,x,y,c) tex2D((tex),(x)*3+(c),(y))
#define tex_get_color(tex,x,y,c) tex2D((tex),((x)*3)+(c),(y))
#define texR(tex,x,y) (((uchar)tex2D((tex),(x),(y)).x))
#define texG(tex,x,y) ((uchar)(tex2D((tex),(x),(y)).y))
#define texB(tex,x,y) ((uchar)(tex2D((tex),(x),(y)).z))

#define cR(c4) (c4.x)
#define cG(c4) (c4.y)
#define cB(c4) (c4.z)

KERNEL void ahd_kernel_interp_g(pixel4* g_horz_res, pixel4* g_vert_res, int width, int height)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x < 2 || y < 2 || x >= width-2 || y >= height-2) {
    return;
  }
  int filter_color = get_filter_color(x,y);

  char4 h_res, v_res;
  /* Copy existing value to output */
  cR(h_res) = cR(v_res) = (filter_color == R) * tex2D(src,x,y);
  cG(h_res) = cG(v_res) = (filter_color == G) * tex2D(src,x,y);
  cB(h_res) = cB(v_res) = (filter_color == B) * tex2D(src,x,y);


  /* Interpolate Green values first */
  if (filter_color == R || filter_color == B) {
    /* Filter color is red or blue Interpolate green channel horizontally */
    /* Use existing green values */
    float sum = (tex2D(src,x-1,y) +
                 tex2D(src,x+1,y))/2.0f;

    /* And use existing red/blue values and apply filter 'h' */
    sum += (-tex2D(src,x-2,y)/4.0f +
             tex2D(src,x,  y)/2.0f +
            -tex2D(src,x+2,y)/4.0f)/4.0f;

    cG(h_res) = (uchar)clampc(sum);

    /* Interpolate green channel vertically */
    /* Use existing green values */
    sum = (tex2D(src,x,y-1) +
           tex2D(src,x,y+1))/2.0f;

    /* And use existing red/blue values and apply filter 'h' */
    sum += (-tex2D(src,x,y-2)/4.0f +
             tex2D(src,x,y  )/2.0f +
            -tex2D(src,x,y+2)/4.0f)/4.0f;

    cG(v_res) = (uchar)clampc(sum);
  }
  int res_index = (y*width + x);
  g_horz_res[res_index] = h_res;
  g_vert_res[res_index] = v_res;
}

KERNEL void ahd_kernel_interp_rb(float4* g_result, pixel *g_tmp_result, int width, int height) {
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  // Take account of padding in source image
  x += P;
  y += P;

  if (x >= width-P || y >= height-P) {
    return;
  }
  pixel pixR = texR(src_g,x,y);
  pixel pixB = texB(src_g,x,y);

  uchar filter_color = get_filter_color(x,y);

  if (filter_color == R || filter_color == B) {
    /* Filter color is red or blue, interpolate missing red or blue channel */
    /* This function operates the same for horiz and vert interpolation */

    int dest_color = (filter_color == R) ? B : R;
    /* Get the difference between the Red/Blue and Green
     * channels */
    float sum =   (-texG(src_g,x-1,y-1)) +
                  (-texG(src_g,x-1,y+1)) +
                  (-texG(src_g,x+1,y-1)) +
                  (-texG(src_g,x+1,y+1));
  if (dest_color == R) {
    sum += texR(src_g,x-1,y-1) +
           texR(src_g,x-1,y+1) +
           texR(src_g,x+1,y-1) +
           texR(src_g,x+1,y+1);
  } else {
    sum += texB(src_g,x-1,y-1) +
           texB(src_g,x-1,y+1) +
           texB(src_g,x+1,y-1) +
           texB(src_g,x+1,y+1);
    }
    /* Apply low pass filter to the difference */
    sum /= 4.0;
    /* Use interpolated or interpolated green value */
    sum += texG(src_g,x,y);
    pixel res = clampc(round(sum));
    if (filter_color == R) {
      pixR = texR(src_g,x,y);
      pixB = res;
    } else {
      pixB = texB(src_g,x,y);
      pixR = res;
    }
    //res_pix[dest_color] = clampc(round(sum));
  } else {
    /* Filter color is green */
    /* Interpolate Red and Blue channels */
    /* This function operates the same for horz and vert interpolation */
    float sum = 0;
    /* Interpolate Red */
    if (even(y)){
      /* Red/Green rows */
      /* Use left and right pixels */
      /* Get the difference between the Red and Green
       * channel (use only the sampled Green values) */
      sum = (texR(src_g,x-1,y) - texG(src_g,x-1,y)) +
            (texR(src_g,x+1,y) - texG(src_g,x+1,y));
    } else {
      /* Blue/Green rows */
      /* Use top and bottom values */
      sum = (texR(src_g,x,y-1) - texG(src_g,x,y-1)) +
            (texR(src_g,x,y+1) - texG(src_g,x,y+1));
    }
    /* Apply low pass filter */
    sum /= 2.0;
    sum += texG(src_g,x,y);
    pixR = clampc(round(sum));;
    //Info("%d,%d Red val %f",x,y,sum);

    /* Interpolate Blue */
    if (odd(y)) {
      /* Blue/Green rows */
      /* Use left and right pixels */
      /* Get the difference between the Red and Green
       * channel (use only the sampled Green values) */
      sum = (texB(src_g,x-1,y) - texG(src_g,x-1,y)) +
            (texB(src_g,x+1,y) - texG(src_g,x+1,y));
    } else {
      /* Red/Green rows */
      /* Use top and bottom values */
      sum = (texB(src_g,x,y-1) - texG(src_g,x,y-1)) +
            (texB(src_g,x,y+1) - texG(src_g,x,y+1));
    }
    /* Apply low pass filter */
    sum /= 2.0;
    sum += texG(src_g,x,y);
    pixB = clampc(round(sum));
    //Info("%d,%d pixB : %d , sum %0.2f G:%d",x,y,pixB,sum,texG(src_g,x,y));
  }

  uint dest_width = width - 2*P;
  int dx = x - P;
  int dy = y - P;


#ifndef _TEST
  if (g_tmp_result != NULL) {
    // During testing, skip global memory access
    pixel *res = get_pix(g_tmp_result,dx,dy,dest_width);
    //pixel *res = g_tmp_result + res_index;
    res[R] = pixR;
    res[G] = texG(src_g,x,y);
    res[B] = pixB;
  }
#endif

  //cuCvRGBtoLab(pixR, pixG, pixB, &res_pix->x, &res_pix->y, &res_pix->z);
  // inlining to avoid passing point arguments

  float4 lab;
  float b = pixB/255.0, r = pixR/255.0;
  float g = texG(src_g,x,y)/255.0;
  float x_, y_, z;

  x_ = b*labXb_32f + g*labXg_32f + r*labXr_32f;
  y_ = b*labYb_32f + g*labYg_32f + r*labYr_32f;
  z =  b*labZb_32f + g*labZg_32f + r*labZr_32f;

  if( x_ > labT_32f )
    x_ = cvCbrt(x_);
  else
    x_ = x_*labSmallScale_32f + labSmallShift_32f;

  if( z > labT_32f )
    z = cvCbrt(z);
  else
    z = z*labSmallScale_32f + labSmallShift_32f;

  if( y_ > labT_32f )
  {
    y_ = cvCbrt(y_);
    lab.x = y_*labLScale_32f - labLShift_32f; // L
  }
  else
  {
    lab.x = y_*labLScale2_32f; // L
    y_ = y_*labSmallScale_32f + labSmallShift_32f;
  }

  lab.y = 500.f*(x_ - y_); // a
  lab.z = 200.f*(y_ - z); // b

  g_result[dx + (dy*dest_width)] = lab;
}

#define BALL_DIST 2
#define HOMO_INNER_LOOP() if (inside(x+dx, y+dy, width, height)) { \
                                  nL = texLs(horz_tex, x+dx, y+dy ); \
                                  na = texAs(horz_tex, x+dx, y+dy ); \
                                  nb = texBs(horz_tex, x+dx, y+dy ); \
                                  lum_diff = diff1(horz.x, nL); \
                                  chrom_diff = diff2(horz.y,horz.z,na,nb); \
                                  if (lum_diff <= lum_thres && \
                                      chrom_diff <= chrom_thres){ \
                                    h_homo++; \
                                  } \
                                  nL = texLs(vert_tex, x+dx, y+dy ); \
                                  na = texAs(vert_tex, x+dx, y+dy ); \
                                  nb = texBs(vert_tex, x+dx, y+dy ); \
                                  lum_diff = diff1(vert.x, nL); \
                                  chrom_diff = diff2(vert.y,vert.z,na,nb); \
                                  if (lum_diff <= lum_thres && \
                                      chrom_diff <= chrom_thres){ \
                                    v_homo++; \
                                  } \
                           }



#define texLs(tex,x,y) ((tex2D((tex),(x),(y)).x))
#define texAs(tex,x,y) ((tex2D((tex),(x),(y)).y))
#define texBs(tex,x,y) ((tex2D((tex),(x),(y)).z))

//#define cR(c4) (c4.x)
//#define cG(c4) (c4.y)
//#define cB(c4) (c4.z)

KERNEL void ahd_kernel_build_homo_map(
    uchar *g_horz_res, uchar* g_vert_res, uint width, uint height/*, int ball_dist*/){
  volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
  volatile int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
      return;
  }

  volatile float4 horz = tex2D(horz_tex,x,y);

  /* Homogenity differences have been calculated for horz and vert directions */
  /* Find the adaptive thresholds, the same threshold is used for horz and vert */
  /* Horizontal case, look at left and right values */
  /* Vertical case, look at top, bottom values */

  /* HORZ */
  // horizontal left and right values

  float lumdiff_1 = diff1(horz.x, texLs(horz_tex, x-1, y));
  float lumdiff_2 = diff1(horz.x, texLs(horz_tex, x+1, y));
  float max_h_lumdiff = MAX(lumdiff_1,lumdiff_2);

  float chromdiff_1 = diff2(horz.y, horz.z,
                     texAs(horz_tex, x-1, y),
                     texBs(horz_tex, x-1, y));

  float chromdiff_2 = diff2(horz.y, horz.z,
                      texAs(horz_tex, x+1, y),
                      texBs(horz_tex, x+1, y));

  float max_h_chromdiff = MAX(chromdiff_1,chromdiff_2);

  volatile float4 vert = tex2D(vert_tex,x,y);

  /* VERT */
  // vertical top and bottom values

  lumdiff_1 = diff1(vert.x, texLs(vert_tex, x, y-1));
  lumdiff_2 = diff1(vert.x, texLs(vert_tex, x, y+1));
  float max_v_lumdiff = MAX(lumdiff_1,lumdiff_2);

  chromdiff_1 = diff2(vert.y, vert.z,
                      texAs(vert_tex, x, y-1),
                      texBs(vert_tex, x, y-1));

  chromdiff_2 = diff2(vert.y, vert.z,
                      texAs(vert_tex, x, y+1),
                      texBs(vert_tex, x, y+1));

  float max_v_chromdiff = MAX(chromdiff_1,chromdiff_2);

  /* THRESHOLD */
  float lum_thres = MIN(max_h_lumdiff,max_v_lumdiff);
  float chrom_thres = MIN(max_h_chromdiff,max_v_chromdiff);


  /* Get the lum and chrom differences for the pixel in the
   * neighbourhood.
   */
  int h_homo = 0;
  int v_homo = 0;

//
//    /* Manual unroll */
//    for (int dy = -BALL_DIST; dy <= BALL_DIST; dy++){
//      for (int dx = -BALL_DIST; dx <= BALL_DIST; dx++) {
//        if (dx == 0 && dy == 0) continue;
//
//            volatile float nL,na,nb,lum_diff,chrom_diff;
//          nL = texLs(horz_tex, x+dx, y+dy);
//          na = texAs(horz_tex, x+dx, y+dy);
//          nb = texBs(horz_tex, x+dx, y+dy);
//
//          lum_diff = diff1(horz.x, nL);
//          chrom_diff = diff2(horz.y,horz.z,na,nb);
//
//            if (lum_diff <= lum_thres &&
//                chrom_diff <= chrom_thres){
//              h_homo++;
//            }
//
//          nL = texLs(vert_tex, x+dx, y+dy);
//          na = texAs(vert_tex, x+dx, y+dy);
//          nb = texBs(vert_tex, x+dx, y+dy);
//
//          lum_diff = diff1(vert.x, nL);
//          chrom_diff = diff2(vert.y,vert.z,na,nb);
//
//            if (lum_diff <= lum_thres &&
//                chrom_diff <= chrom_thres){
//              v_homo++;
//            }
//
//      }
//    }

  float chrom_diff,lum_diff,na,nb,nL;
  int dx,dy;
  dx = -2; dy = -2; HOMO_INNER_LOOP();
  dx = -1; dy = -2; HOMO_INNER_LOOP();
  dx = 0; dy = -2; HOMO_INNER_LOOP();
  dx = 1; dy = -2; HOMO_INNER_LOOP();
  dx = 2; dy = -2; HOMO_INNER_LOOP();
  dx = -2; dy = -1; HOMO_INNER_LOOP();
  dx = -1; dy = -1; HOMO_INNER_LOOP();
  dx = 0; dy = -1; HOMO_INNER_LOOP();
  dx = 1; dy = -1; HOMO_INNER_LOOP();
  dx = 2; dy = -1; HOMO_INNER_LOOP();
  dx = -2; dy = 0; HOMO_INNER_LOOP();
  dx = -1; dy = 0; HOMO_INNER_LOOP();
  dx = 1; dy = 0; HOMO_INNER_LOOP();
  dx = 2; dy = 0; HOMO_INNER_LOOP();
  dx = -2; dy = 1; HOMO_INNER_LOOP();
  dx = -1; dy = 1; HOMO_INNER_LOOP();
  dx = 0; dy = 1; HOMO_INNER_LOOP();
  dx = 1; dy = 1; HOMO_INNER_LOOP();
  dx = 2; dy = 1; HOMO_INNER_LOOP();
  dx = -2; dy = 2; HOMO_INNER_LOOP();
  dx = -1; dy = 2; HOMO_INNER_LOOP();
  dx = 0; dy = 2; HOMO_INNER_LOOP();
  dx = 1; dy = 2; HOMO_INNER_LOOP();
  dx = 2; dy = 2; HOMO_INNER_LOOP();

//    char4 result;
//    result.x = clampc(h_homo);
//    result.y = clampc(v_homo);
//    g_homo_res[x+y*width] = result;
  *(get_homo(g_horz_res,x,y,width)) = h_homo;
  *(get_homo(g_vert_res,x,y,width)) = v_homo;
}

KERNEL void ahd_kernel_choose_direction(pixel *g_result, float *g_direction, uint width, uint height) {
  volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
  volatile int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
      return;
  }
  int horz_score = 0;
  int vert_score = 0;
  for (int dy = -1; dy <= 1; dy++){
    for (int dx = -1; dx <= 1; dx++) {
        // todo divide the score by the area so that this
        // works properly at the borders
        horz_score += tex2D(homo_h_tex,x+dx,y+dy);
        vert_score += tex2D(homo_v_tex,x+dx,y+dy);

    }
  }

  if (g_direction) {
      if (vert_score > horz_score) {// ? VERT : HORZ;
        *get_homo(g_direction,x,y,width) = 0;
      } else {
        *get_homo(g_direction,x,y,width) = 255;
      }
  }

  float L,a,b;
  if (vert_score <= horz_score) {
     L = texLs(horz_tex,x,y);
     a = texAs(horz_tex,x,y);
     b = texBs(horz_tex,x,y);
  } else {
    L = texLs(vert_tex,x,y);
    a = texAs(vert_tex,x,y);
    b = texBs(vert_tex,x,y);
  }
  pixel *res = get_pix(g_result,x,y,width);
  cuCvLabtoRGB(L,a,b,res+R,res+G,res+B);
}



DEVICE int median3(int *a, int n) {
  /* Perform insertion sort */
  for (int i = 1; i<n; i++) {
    int val = a[i];
    int j = i;
    while (j > 0 && a[j-1] > val) {
      a[j] = a[j-1];
      j--;
    }
    a[j] = val;
  }
  /* Return median value */
  return a[n/2];
}

#define SORT(a,b) { if ((a)>(b)) SWAPV((a),(b)); }
#define SWAPV(a,b) { int temp=(a);(a)=(b);(b)=temp; }
DEVICE int median4(int * p) {
    SORT(p[1], p[2]) ; SORT(p[4], p[5]) ; SORT(p[7], p[8]) ;
    SORT(p[0], p[1]) ; SORT(p[3], p[4]) ; SORT(p[6], p[7]) ;
    SORT(p[1], p[2]) ; SORT(p[4], p[5]) ; SORT(p[7], p[8]) ;
    SORT(p[0], p[3]) ; SORT(p[5], p[8]) ; SORT(p[4], p[7]) ;
    SORT(p[3], p[6]) ; SORT(p[1], p[4]) ; SORT(p[2], p[5]) ;
    SORT(p[4], p[7]) ; SORT(p[4], p[2]) ; SORT(p[6], p[4]) ;
    SORT(p[4], p[2]) ; return(p[4]) ;
}


DEVICE int cu_intcmp(const void *p1, const void *p2) {
  int a = *((int *)p1);
  int b = *((int *)p2);
  return a-b;
}

DEVICE int median_diff(int x, int y, int chan1, int chan2) {
  int diffs[9];
  pixel val1,val2;
  /* Manual unroll */
  /* Avoids local memory use */
//    for (int dy = -1; dy <= 1; dy++) {
//      for (int dx = -1; dx <= 1; dx++) {
//        //printf("dx = %d; dy = %d;\n",dx,dy);
//        pixel val1 = tex_get_color(src,x+dx,y+dy,chan1);
//        pixel val2 = tex_get_color(src,x+dx,y+dy,chan2);
//        diffs[i] = val1 - val2;
//        i++;
//      }
//    }

  val1 = tex_get_color(src,x+-1,y+-1,chan1);
  val2 = tex_get_color(src,x+-1,y+-1,chan2);
  diffs[0] = val1 - val2;

  val1 = tex_get_color(src,x,y-1,chan1);
  val2 = tex_get_color(src,x,y-1,chan2);
  diffs[1] = val1 - val2;

  val1 = tex_get_color(src,x+1,y+-1,chan1);
  val2 = tex_get_color(src,x+1,y+-1,chan2);
  diffs[2] = val1 - val2;

  val1 = tex_get_color(src,x+-1,y,chan1);
  val2 = tex_get_color(src,x+-1,y,chan2);
  diffs[3] = val1 - val2;

  val1 = tex_get_color(src,x,y,chan1);
  val2 = tex_get_color(src,x,y,chan2);
  diffs[4] = val1 - val2;

  val1 = tex_get_color(src,x+1,y,chan1);
  val2 = tex_get_color(src,x+1,y,chan2);
  diffs[5] = val1 - val2;

  val1 = tex_get_color(src,x+-1,y+1,chan1);
  val2 = tex_get_color(src,x+-1,y+1,chan2);
  diffs[6] = val1 - val2;

  val1 = tex_get_color(src,x,y+1,chan1);
  val2 = tex_get_color(src,x,y+1,chan2);
  diffs[7] = val1 - val2;

  val1 = tex_get_color(src,x+1,y+1,chan1);
  val2 = tex_get_color(src,x+1,y+1,chan2);
  diffs[8] = val1 - val2;

  //int m = median3(diffs,i); /* insertion sort */
  int m = median4(diffs); /* network sort */
  return m;
}

KERNEL void ahd_kernel_remove_artefacts (pixel *g_result, uint width, uint height){
    volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    //volatile pixel *dest = get_pix(g_result,x,y,width);

    int res = median_diff(x,y,R,G) + tex_get_color(src,x,y,G);
    g_result[(x+y*width)*3+R] = clampc(res);

    res = median_diff(x,y,B,G) + tex_get_color(src,x,y,G);
    g_result[(x+y*width)*3+B] = clampc(res);

    res = round((median_diff(x,y,G,R) +
             median_diff(x,y,G,B) +
             tex_get_color(src,x,y,R) +
             tex_get_color(src,x,y,B))/2.0);
    g_result[(x+y*width)*3+G] = clampc(res);
}
