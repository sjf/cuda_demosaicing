/*
 * colorspace.c
 *
 *  Created on: 11 Mar 2010
 *      Author: sjf
 *
 *  Based on pseudo code from http://www.easyrgb.com/index.php?X=MATH&H=01
 *
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "colorspace.h"
#include "base.h"


void rgb_to_cielab(unsigned char r, unsigned char g, unsigned char b,
    float *L, float *a, float *b_) {

  float var_R = r/255.0;        //R from 0 to 255
  float var_G = g/255.0;        //G from 0 to 255
  float var_B = b/255.0;        //B from 0 to 255

  if ( var_R > 0.04045 ){
    var_R = pow(( var_R + 0.055 ) / 1.055,  2.4);
  } else {
    var_R = var_R / 12.92;
  }
  if ( var_G > 0.04045 ) {
    var_G = pow(( var_G + 0.055 ) / 1.055, 2.4);
  } else {
    var_G = var_G / 12.92;
  }
  if ( var_B > 0.04045 ) {
    var_B = pow(( var_B + 0.055 ) / 1.055, 2.4);
  } else {
    var_B = var_B / 12.92;
  }

  var_R = var_R * 100;
  var_G = var_G * 100;
  var_B = var_B * 100;

  //Observer. = 2°, Illuminant = D65
  float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
  float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
  float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

  /*printf("RGB -> XYZ: %f,%f,%f\n",X,Y,Z);*/

  float var_X = X / 95.047;  //ref_X =  95.047   Observer= 2°, Illuminant= D65
  float var_Y = Y / 100.0;     //ref_Y = 100.000
  float var_Z = Z / 108.883; //ref_Z = 108.883

  if ( var_X > 0.008856 ) {
    var_X = pow(var_X, 1.0/3.0);
  } else {
    var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 );
  }
  if ( var_Y > 0.008856 ) {
    var_Y = pow(var_Y, 1.0/3.0);
  } else {
    var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 );
  }
  if ( var_Z > 0.008856 ) {
    var_Z = pow(var_Z, 1.0/3.0);
  } else {
    var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 );
  }

  *L = ( 116.0 * var_Y ) - 16.0;
  *a = 500.0 * ( var_X - var_Y );
  *b_ = 200.0 * ( var_Y - var_Z );
}

/****************************************************************************************\
*                                     RGB <-> L*a*b*                                     *
\****************************************************************************************/

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

#define  labT   fix(labT_32f*255,lab_shift)

#undef lab_shift
#define lab_shift 10
#define labXr  fix(labXr_32f,lab_shift)
#define labXg  fix(labXg_32f,lab_shift)
#define labXb  fix(labXb_32f,lab_shift)

#define labYr  fix(labYr_32f,lab_shift)
#define labYg  fix(labYg_32f,lab_shift)
#define labYb  fix(labYb_32f,lab_shift)

#define labZr  fix(labZr_32f,lab_shift)
#define labZg  fix(labZg_32f,lab_shift)
#define labZb  fix(labZb_32f,lab_shift)

#define labSmallScale_32f  7.787f
#define labSmallShift_32f  0.13793103448275862f  /* 16/116 */
#define labLScale_32f      116.f
#define labLShift_32f      16.f
#define labLScale2_32f     903.3f

#define labSmallScale fix(31.27 /* labSmallScale_32f*(1<<lab_shift)/255 */,lab_shift)
#define labSmallShift fix(141.24138 /* labSmallScale_32f*(1<<lab) */,lab_shift)
#define labLScale     fix(295.8 /* labLScale_32f*255/100 */,lab_shift)
#define labLShift     fix(41779.2 /* labLShift_32f*1024*255/100 */,lab_shift)
#define labLScale2    fix(labLScale2_32f*0.01,lab_shift)

#define fix(x,n)      (int)((x)*(1 << (n)) + 0.5)
#define descale(x,n)  (((x) + (1 << ((n)-1))) >> (n))
#define CV_DESCALE    descale
#define CV_CAST_8U(t) (unsigned char)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)

/* 1024*(([0..511]./255)**(1./3)) */
static unsigned short icvLabCubeRootTab[] = {
       0,  161,  203,  232,  256,  276,  293,  308,  322,  335,  347,  359,  369,  379,  389,  398,
     406,  415,  423,  430,  438,  445,  452,  459,  465,  472,  478,  484,  490,  496,  501,  507,
     512,  517,  523,  528,  533,  538,  542,  547,  552,  556,  561,  565,  570,  574,  578,  582,
     586,  590,  594,  598,  602,  606,  610,  614,  617,  621,  625,  628,  632,  635,  639,  642,
     645,  649,  652,  655,  659,  662,  665,  668,  671,  674,  677,  680,  684,  686,  689,  692,
     695,  698,  701,  704,  707,  710,  712,  715,  718,  720,  723,  726,  728,  731,  734,  736,
     739,  741,  744,  747,  749,  752,  754,  756,  759,  761,  764,  766,  769,  771,  773,  776,
     778,  780,  782,  785,  787,  789,  792,  794,  796,  798,  800,  803,  805,  807,  809,  811,
     813,  815,  818,  820,  822,  824,  826,  828,  830,  832,  834,  836,  838,  840,  842,  844,
     846,  848,  850,  852,  854,  856,  857,  859,  861,  863,  865,  867,  869,  871,  872,  874,
     876,  878,  880,  882,  883,  885,  887,  889,  891,  892,  894,  896,  898,  899,  901,  903,
     904,  906,  908,  910,  911,  913,  915,  916,  918,  920,  921,  923,  925,  926,  928,  929,
     931,  933,  934,  936,  938,  939,  941,  942,  944,  945,  947,  949,  950,  952,  953,  955,
     956,  958,  959,  961,  962,  964,  965,  967,  968,  970,  971,  973,  974,  976,  977,  979,
     980,  982,  983,  985,  986,  987,  989,  990,  992,  993,  995,  996,  997,  999, 1000, 1002,
    1003, 1004, 1006, 1007, 1009, 1010, 1011, 1013, 1014, 1015, 1017, 1018, 1019, 1021, 1022, 1024,
    1025, 1026, 1028, 1029, 1030, 1031, 1033, 1034, 1035, 1037, 1038, 1039, 1041, 1042, 1043, 1044,
    1046, 1047, 1048, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1065,
    1066, 1067, 1068, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1084,
    1085, 1086, 1088, 1089, 1090, 1091, 1092, 1094, 1095, 1096, 1097, 1098, 1099, 1101, 1102, 1103,
    1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1117, 1118, 1119, 1120, 1121,
    1122, 1123, 1124, 1125, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1138, 1139,
    1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1154, 1155, 1156,
    1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
    1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204,
    1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1215, 1216, 1217, 1218, 1219,
    1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1230, 1231, 1232, 1233, 1234,
    1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
    1250, 1251, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1259, 1260, 1261, 1262, 1263,
    1264, 1265, 1266, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1273, 1274, 1275, 1276, 1277,
    1278, 1279, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1285, 1286, 1287, 1288, 1289, 1290, 1291
};

void icvBGRx2Lab_8u_CnC3R( unsigned char r1, unsigned char g1, unsigned char b1,
        float *L_, float *a_, float *b_){
    int b = b1, g = g1, r = r1;
    int x, y, z, f;
    int L, a;

    x = b*labXb + g*labXg + r*labXr;
    y = b*labYb + g*labYg + r*labYr;
    z = b*labZb + g*labZg + r*labZr;

    f = x > labT;
    x = CV_DESCALE( x, lab_shift );

    if( f )
        assert( (unsigned)x < 512 ), x = icvLabCubeRootTab[x];
    else
        x = CV_DESCALE(x*labSmallScale + labSmallShift,lab_shift);

    f = z > labT;
    z = CV_DESCALE( z, lab_shift );

    if( f )
        assert( (unsigned)z < 512 ), z = icvLabCubeRootTab[z];
    else
        z = CV_DESCALE(z*labSmallScale + labSmallShift,lab_shift);

    f = y > labT;
    y = CV_DESCALE( y, lab_shift );

    if( f )
    {
        assert( (unsigned)y < 512 ), y = icvLabCubeRootTab[y];
        L = CV_DESCALE(y*labLScale - labLShift, 2*lab_shift );
    }
    else
    {
        L = CV_DESCALE(y*labLScale2,lab_shift);
        y = CV_DESCALE(y*labSmallScale + labSmallShift,lab_shift);
    }

    a = CV_DESCALE( 500*(x - y), lab_shift ) + 128;
    b = CV_DESCALE( 200*(y - z), lab_shift ) + 128;

    *L_ = CV_CAST_8U(L);
    *a_ = CV_CAST_8U(a);
    *b_ = CV_CAST_8U(b);
}

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
}
Cv32suf;


#define cvCbrt cubeRoot

float  cubeRoot( float value )
{
  // Unoptimised code:
  //return pow(value,1.0/3.0);

    float fr;
    Cv32suf v, m;
    int ix, s;
    int ex, shx;

    v.f = value;
    ix = v.i & 0x7fffffff;
    s = v.i & 0x80000000;
    ex = (ix >> 23) - 127;
    shx = ex % 3;
    shx -= shx >= 0 ? 3 : 0;
    ex = (ex - shx) / 3; /* exponent of cube root */
    v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
    fr = v.f;

    /* 0.125 <= fr < 1.0 */
    /* Use quartic rational polynomial with error < 2^(-24) */
    fr = (float)(((((45.2548339756803022511987494 * fr +
    192.2798368355061050458134625) * fr +
    119.1654824285581628956914143) * fr +
    13.43250139086239872172837314) * fr +
    0.1636161226585754240958355063)/
    ((((14.80884093219134573786480845 * fr +
    151.9714051044435648658557668) * fr +
    168.5254414101568283957668343) * fr +
    33.9905941350215598754191872) * fr +
    1.0));

    /* fr *= 2^ex * sign */
    m.f = value;
    v.f = fr;
    v.i = (v.i + (ex << 23) + s) & (m.i*2 != 0 ? -1 : 0);
    return v.f;
}

void cvRGBtoLab( unsigned char r1, unsigned char g1, unsigned char b1,
        float *L_, float *a_, float *b_)
{

  //float b = b1, g = g1, r = r1;
  float b = b1/255.0, g = g1/255.0, r = r1/255.0;
    float x, y, z;
    float L, a;

    x = b*labXb_32f + g*labXg_32f + r*labXr_32f;
    y = b*labYb_32f + g*labYg_32f + r*labYr_32f;
    z = b*labZb_32f + g*labZg_32f + r*labZr_32f;

    if( x > labT_32f )
        x = cvCbrt(x);
    else
        x = x*labSmallScale_32f + labSmallShift_32f;

    if( z > labT_32f )
        z = cvCbrt(z);
    else
        z = z*labSmallScale_32f + labSmallShift_32f;

    if( y > labT_32f )
    {
        y = cvCbrt(y);
        L = y*labLScale_32f - labLShift_32f;
    }
    else
    {
        L = y*labLScale2_32f;
        y = y*labSmallScale_32f + labSmallShift_32f;
    }

    a = 500.f*(x - y);
    b = 200.f*(y - z);

    *L_ = L;
    *a_ = a;
    *b_ = b;
}

void cvLabtoRGB( float L, float a, float b,
    unsigned char *r_, unsigned char *g_, unsigned char *b_/*, float *R, float *G, float *B */) {

  float x, y, z;
  float g, r;

  L = (L + labLShift_32f)*(1.f/labLScale_32f);
  x = (L + a*0.002f);
  z = (L - b*0.005f);
  y = L*L*L;
  x = x*x*x;
  z = z*z*z;


  b = x*labBx_32f + y*labBy_32f + z*labBz_32f;
  g = x*labGx_32f + y*labGy_32f + z*labGz_32f;
  r = x*labRx_32f + y*labRy_32f + z*labRz_32f;

  //*R = r; *G = g; *B = b;
  //printf("r,g,b %f,%f,%f\n" ,r,g,b);

  *r_ = ROUND(r*255);
  *g_ = ROUND(g*255);
  *b_ = ROUND(b*255);
}



void cielab_to_rgb(float L, float a, float b_,
    unsigned char *r, unsigned char *g, unsigned char *b) {
  float var_Y = ( L + 16.0 ) / 116.0;
  float var_X = a / 500.0 + var_Y;
  float var_Z = var_Y - b_ / 200.0;

  float y3 = pow(var_Y,3.0);
  float x3 = pow(var_X,3.0);
  float z3 = pow(var_Z,3.0);
  if ( y3 > 0.008856 ) {
    var_Y = y3;
  }
  else {
    var_Y = ( var_Y - 16.0 / 116.0 ) / 7.787;
  }
  if ( x3 > 0.008856 ) {
    var_X = x3;
  } else {
    var_X = ( var_X - 16.0 / 116.0 ) / 7.787;
  }
  if ( z3 > 0.008856 ) {
    var_Z = z3;
  } else {
    var_Z = ( var_Z - 16.0 / 116.0 ) / 7.787;
  }

  float X = 95.047 * var_X;     //ref_X =  95.047     Observer= 2°, Illuminant= D65
  float Y = 100.0 * var_Y;     //ref_Y = 100.000
  float Z = 108.883 * var_Z;     //ref_Z = 108.883

  /*printf("Lab -> XYZ: %f,%f,%f\n",X,Y,Z);*/

  var_X = X / 100.0;        //X from 0 to  95.047      (Observer = 2°, Illuminant = D65)
  var_Y = Y / 100.0;        //Y from 0 to 100.000
  var_Z = Z / 100.0;        //Z from 0 to 108.883

  float var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
  float var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
  float var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

  if ( var_R > 0.0031308 ) {
    var_R = 1.055 *  pow(var_R, 1/2.4) - 0.055;
  } else {
    var_R = 12.92 * var_R;
  }
  if ( var_G > 0.0031308 ) {
    var_G = 1.055 * pow(var_G, 1/2.4) - 0.055;
  } else {
    var_G = 12.92 * var_G;
  }
  if ( var_B > 0.0031308 ) {
    var_B = 1.055 * pow(var_B, 1/2.4) - 0.055;
  }
  else {
    var_B = 12.92 * var_B;
  }

  *r = round(var_R * 255.0);
  *g = round(var_G * 255.0);
  *b = round(var_B * 255.0);
}


#ifdef _COL_TEST
int main(){
#else
int test(){
#endif
  int r,g,b;
  unsigned char r1,g1,b1;
  unsigned char r2,g2,b2;
  float L,a,bs;
  float L1,a1,bs1;
//  float R_,G_,B_;

  for (r = 0; r < 255; r+=10) {
    for (g = 0; g < 255; g+=10) {
      for (b = 0; b < 255; b+=10) {
        rgb_to_cielab(r,g,b,&L,&a,&bs);
        cvRGBtoLab(r,g,b,&L1,&a1,&bs1);


        cielab_to_rgb(L,a,bs,&r1,&g1,&b1);
        cvLabtoRGB(L1,a1,bs1,&r2,&g2,&b2/*,&R,&G,&B*/);

        if (r != r2 || g != g2 || b != b2) {
          printf("OpenCV  (%d,%d,%d) -> %0.4f,%0.4f,%0.4f -> (%d,%d,%d)\n",
                  r,g,b, L1,a1,bs1, /*R_,G_,B_,*/ r2,g2,b2);
          printf("EasyRGB (%d,%d,%d) -> %0.4f,%0.4f,%0.4f\n",r,g,b,L,a,bs);
        }
        assert(r == r2+1 ||r == r2-1 || r == r2);
        assert(g == g2+1 ||g == g2-1 || g == g2);
        assert(b == b2+1 ||b == b2-1 || b == b2);
      }
    }
  }
  return 0;
}

