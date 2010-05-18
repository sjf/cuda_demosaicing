/*
 * bayer.c
 *
 *  Created on: 27 Jan 2010
 *      Author: sjf65
 */

#include "base.h"
#include "image.h"
#include "bayer.h"

const static char * bayer_name[] = {
    "RGGB", "GRBG", "BGGR", "GBRG", "Unsupported Bayer"};
const char *get_bayer_name(int num) {
  if (num < 0 || num >= 4) {
    return bayer_name[BAYER_UNK];
  }
  return bayer_name[num];
}
