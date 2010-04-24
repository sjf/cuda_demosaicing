/*
 * bayer.h
 *
 *  Created on: 27 Jan 2010
 *      Author: sjf65
 */

#ifndef BAYER_H_
#define BAYER_H_

#include "image.h"

#define RGGB 0
#define GRBG 1
#define BGGR 2
#define GBRG 3
#define BAYER_UNK 4

#define  odd(n) ((n)&1)
#define even(n) (!odd((n)))


/*
 * Bayer Pattern:
 * RGRG
 * GBGB
 */

//const static int bayer[2][2] = {{R,G},{G,B}};
const static uchar bayer[4][2][2] =
 {{{R,G},{G,B}},
  {{G,R},{B,G}},
  {{B,G},{G,R}},
  {{G,B},{R,G}}};

// CUDA only supports bayer type 0: RGGB
#ifdef __CUDACC__
#define get_filter_color(x,y) (even(x) ? (even(y) ? R : G) : (odd(y) ? B : G))
#else
#define get_filter_color(x,y) (bayer[settings->bayer_type][((int)y)&1][(x)&1] )
#endif

//#define get_filter_color(x,y) (bayer[(y)&1][(x)&1] )

//#define isred(x,y)  (even(y) && even(x))
//#define isblue(x,y) (odd(y) && odd(x))
//#define isgreen(x,y) ((even(y) && odd(x)) || (odd(y) && even(x)))

const char *get_bayer_name(int num);

#endif /* BAYER_H_ */
