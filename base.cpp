/*
 * base.c
 *
 *  Created on: 28 Jan 2010
 *      Author: sjf65
 */

#include "base.h"
#include "time.h"
#include "bayer.h"

char *time_str(){
  time_t t = time(NULL);
  char *res = ctime(&t);
  // remove newline
  res[strlen(res)-1] = '\0';
  return res;
}

void init_settings(settings_t *settings){
  bzero(settings,sizeof(settings_t));
  settings->use_ahd = 1;
  settings->median_filter_iterations = 3;
  settings->save_temps = 1;
  settings->ball_distance = 2;
  settings->dilations = 1;
  settings->edge_threshold = THRESHOLD;
}

void init_image_settings(char *image_name) {
    settings->image_name = image_name;
    settings->bayer_type = RGGB;
    settings->mask_size = 0;
}

void *callocz(size_t nmemb, size_t size){
    void *buf = calloc(nmemb, size);
    bzero(buf,nmemb*size);
    return buf;
}

void out_filename(const char *filename, char *buf, size_t buf_len) {
    uint len = strlen(filename);
  const char *exten = NULL;
  if (settings->use_ahd) {
    if (settings->ahd_mask) {
      exten = (char *)malloc(len + 100);
      sprintf((char *)exten,"ahdmask%d.ppm",settings->ahd_mask);
    } else if (settings->use_cuda) {
      exten = "ahd_cuda.ppm";
    } else {
      exten = "ahd.ppm";
    }
  }
  else {
    exten = (settings->use_cuda) ? "bil_cuda.ppm" : "bilin.ppm";
  }

    assert(buf_len >= len -3 + strlen(exten));
    strncpy(buf,filename,len-3); // copy filename excluding extension
    strcat(buf,exten); // append new extension
    assert(strlen(buf) == len-3+strlen(exten)); // check everything worked
}

int intcmp(const void *p1, const void *p2) {
  int a = *((int *)p1);
  int b = *((int *)p2);
  return a-b;
}
