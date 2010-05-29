/*
 * colorspace.cu
 *
 *  Created on: 26 Mar 2010
 *      Author: sjf
 */

#define  labXr_32f  0.433953f /* = xyzXr_32f / 0.950456 */
#define  labXg_32f  0.376219f /* = xyzXg_32f / 0.950456 */
#define  labXb_32f  0.189828f /* = xyzXb_32f / 0.950456 */

#define  labYr_32f  0.212671f /* = xyzYr_32f */
#define  labYg_32f  0.715160f /* = xyzYg_32f */
#define  labYb_32f  0.072169f /* = xyzYb_32f */

#define  labZr_32f  0.017758f /* = xyzZr_32f / 1.088754 */
#define  labZg_32f  0.109477f /* = xyzZg_32f / 1.088754 */
#define  labZb_32f  0.872766f /* = xyzZb_32f / 1.088754 */

#define  labRx_32f  3.0799327f  /* = xyzRx_32f * 0.950456 */
#define  labRy_32f  (-1.53715f) /* = xyzRy_32f */
#define  labRz_32f  (-0.542782f)/* = xyzRz_32f * 1.088754 */

#define  labGx_32f  (-0.921235f)/* = xyzGx_32f * 0.950456 */
#define  labGy_32f  1.875991f   /* = xyzGy_32f */
#define  labGz_32f  0.04524426f /* = xyzGz_32f * 1.088754 */

#define  labBx_32f  0.0528909755f /* = xyzBx_32f * 0.950456 */
#define  labBy_32f  (-0.204043f)  /* = xyzBy_32f */
#define  labBz_32f  1.15115158f   /* = xyzBz_32f * 1.088754 */

#define  labT_32f   0.008856f

//#define labT   fix(labT_32f*255,lab_shift)
//
//#undef lab_shift
//#define lab_shift 10
//#define labXr  fix(labXr_32f,lab_shift)
//#define labXg  fix(labXg_32f,lab_shift)
//#define labXb  fix(labXb_32f,lab_shift)
//
//#define labYr  fix(labYr_32f,lab_shift)
//#define labYg  fix(labYg_32f,lab_shift)
//#define labYb  fix(labYb_32f,lab_shift)
//
//#define labZr  fix(labZr_32f,lab_shift)
//#define labZg  fix(labZg_32f,lab_shift)
//#define labZb  fix(labZb_32f,lab_shift)

#define labSmallScale_32f  7.787f
#define labSmallShift_32f  0.13793103448275862f  /* 16/116 */
#define labLScale_32f      116.f
#define labLShift_32f      16.f
#define labLScale2_32f     903.3f

//#define labSmallScale fix(31.27 /* labSmallScale_32f*(1<<lab_shift)/255 */,lab_shift)
//#define labSmallShift fix(141.24138 /* labSmallScale_32f*(1<<lab) */,lab_shift)
//#define labLScale fix(295.8 /* labLScale_32f*255/100 */,lab_shift)
//#define labLShift fix(41779.2 /* labLShift_32f*1024*255/100 */,lab_shift)
//#define labLScale2 fix(labLScale2_32f*0.01,lab_shift)

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
}
Cv32suf;

#define cvCbrt cuCubeRoot

DEVICE float  cuCubeRoot( float value )
{
  return __powf(value,1.0f/3.0f);
//
//    float fr;
//    Compiling for CUDA, m and v are put in local memory
//    Change it to use frexp instead of the union thingy
//    Cv32suf v, m;
//    int ix, s;
//    int ex, shx;
//
//    v.f = value;
//    ix = v.i & 0x7fffffff;
//    s = v.i & 0x80000000;
//    ex = (ix >> 23) - 127;
//    shx = ex % 3;
//    shx -= shx >= 0 ? 3 : 0;
//    ex = (ex - shx) / 3; /* exponent of cube root */
//    v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
//    fr = v.f;
//
//    /* 0.125 <= fr < 1.0 */
//    /* Use quartic rational polynomial with error < 2^(-24) */
//    fr = (float)(((((45.2548339756803022511987494 * fr +
//    192.2798368355061050458134625) * fr +
//    119.1654824285581628956914143) * fr +
//    13.43250139086239872172837314) * fr +
//    0.1636161226585754240958355063)/
//    ((((14.80884093219134573786480845 * fr +
//    151.9714051044435648658557668) * fr +
//    168.5254414101568283957668343) * fr +
//    33.9905941350215598754191872) * fr +
//    1.0));
//
//    /* fr *= 2^ex * sign */
//    m.f = value;
//    v.f = fr;
//    v.i = (v.i + (ex << 23) + s) & (m.i*2 != 0 ? -1 : 0);
//    return v.f;
}

DEVICE void cuCvRGBtoLab( unsigned char pixR, unsigned char pixG, unsigned char pixB,
        float4 *g_result)
{
   float4 lab;
  float b = pixB/255.0, g = pixG/255.0, r = pixR/255.0;
  float x_, y_, z;

  x_ = b*labXb_32f + g*labXg_32f + r*labXr_32f;
  y_ = b*labYb_32f + g*labYg_32f + r*labYr_32f;
  z  = b*labZb_32f + g*labZg_32f + r*labZr_32f;

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

  *g_result = lab;
}

DEVICE void cuCvLabtoRGB( float L, float a, float b,
    unsigned char *r_, unsigned char *g_, unsigned char *b_) {

  float x, y, z;

  L = (L + labLShift_32f)*(1.f/labLScale_32f);
  x = (L + a*0.002f);
  z = (L - b*0.005f);
  y = L*L*L;
  x = x*x*x;
  z = z*z*z;

  float g, r;
  b = x*labBx_32f + y*labBy_32f + z*labBz_32f;
  *b_ = ROUND(b*255);

  g = x*labGx_32f + y*labGy_32f + z*labGz_32f;
  *g_ = ROUND(g*255);

  r = x*labRx_32f + y*labRy_32f + z*labRz_32f;
  *r_ = ROUND(r*255);


}
