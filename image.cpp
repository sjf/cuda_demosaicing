/*
 * image.c
 *
 *  Created on: 28 Jan 2010
 *      Author: sjf65
 */
#include <math.h>

#include "base.h"
#include "image.h"
#include "sys/times.h"
#include "util.h"
#include "bayer.h"

#define BUF_SIZE 1024
#define MIN_HDR_LEN 10
#define MAX_DEPTH 65535

img *new_image(uint height, uint width) {
    img *image = mallocz<img>(sizeof(img));
    image->height = height;
    image->width = width;
    return image;
}

img *new_image(uint height, uint width, pixel *buffer) {
    img *image = mallocz<img>(sizeof(img));
    image->height = height;
    image->width = width;
    if (buffer == NULL) {
      buffer = mallocz<pixel>(height * width * RGB);
    }
    image->buffer = buffer;
    return image;
}


img *read_image_from_file(char *filename){
  img *result = NULL;

  FILE *fh = fopen(filename,"r");
  if (!fh) {
    FatalSysError("Could not open '%s'", filename);
  }
  char buf[1024];
  if (fscanf(fh,"%1024s",buf) < 1) {
    FatalError("'%s' is not a valid PPM file", filename);
  }
  if (strcmp(buf,"P6") != 0) {
    FatalError("'%s' is not a PPM file", filename);
  }
  /* Read comments */
  int res = fscanf(fh," #%1024[^\n\r]",buf);
  if (res < 0) {
    FatalError("Error reading PPM file '%s'",filename);
  }
  /* Look for bayer comment hint */
  if (res > 0) {
    if (strstr(buf,        "RGGB")) {
      settings->bayer_type = RGGB;
    } else if (strstr(buf, "GRBG")) {
      settings->bayer_type = GRBG;
    } else if (strstr(buf, "BGGR")) {
      settings->bayer_type = BGGR;
    } else if (strstr(buf, "GBRG")) {
      settings->bayer_type = GBRG;
    } else {
      //Warn("No bayer hint in comment '%s'",buf);
    }
  }

  int height, width, depth;
  height = width = depth = -1;

  /* width, height, depth followed by exactly 1 whitespace character */
  res = fscanf(fh,"%d %d %d%1[\n\t ]",&width,&height,&depth,buf);
  if (res < 0) {
    FatalSysError("Error reading width, height, depth from PPM file '%s'", filename);
  }
  if (res < 3) {
    FatalError("Could not read width, height, depth from PPM file '%s'", filename);
  }
  if (depth <= 0 || depth > MAX_DEPTH) {
    FatalError("Error reading PPM file '%s', invalid color depth",filename);
  }
  //uint pix_width = (depth <= 255) ? 1 : 2;
  if (depth > 255) {
    FatalError("Colour depth %d not supported", depth);
  }
  Info("Loading PPM image %dx%d depth %d",width,height,depth);

  int buf_size = height*width*RGB;
  uchar *buffer = mallocz<uchar>(buf_size);
  res = fread(buffer,sizeof(uchar),buf_size,fh);
  if (res < 0) {
    FatalSysError("Error reading PPM file '%s'",filename);
  }
  if (res < buf_size) {
    FatalError("Error reading PPM file '%s', file too short, read %d bytes, should be %d", filename, res, buf_size);
  }
  result = new_image(height,width);
  result->buffer = buffer;
  return result;
}

int write_result_to_file(const char *orig_name, img *image){
  size_t len = strlen(orig_name) + 10;
  char *out_name = malloczero<char>(len);
  out_filename(orig_name,out_name,len);

  write_image_to_file(image->buffer, image->width, image->height, (const char*)out_name);

  Info("Wrote image to %s",out_name);
  free(out_name);
  return 1;
}

int write_image_to_file(pixel *buf, uint width, uint height, const char *filename) {
  FILE *fh = fopen(filename,"w");
  if (!fh) {
    FatalSysError("Could not open '%s'", filename);
  }
  uint depth = 255;

  fprintf(fh,"P6\n# ");
  fprintf(fh, "Created: %s ",time_str());
  fprintf(fh, "Executed on: %s ", settings->use_cuda ? "CUDA" : "Host");
  if (settings->use_cuda) {
#ifdef __CUDAEMU__
    fprintf(fh, "(Emulation)  ");
#endif
  }
  fprintf(fh, "Method: %s ", settings->use_ahd ? "AHD" : "Linear");
  if (settings->use_ahd) {
    fprintf(fh, "ball_dist: %d ", settings->ball_distance);
    fprintf(fh, "no. median filters: %d ", settings->median_filter_iterations);
  }
#ifdef _TEST
  fprintf(fh, "compiled with _TEST and -O3  ");
#else
  fprintf(fh, "compiled withOUT _TEST ");
#endif

  if (fprintf(fh, "\n%d %d\n%d\n", width, height, depth) < 0) {
    FatalSysError("Error writing to file '%s'",filename);
  }
  uint size = width*height*RGB;
  if (fwrite(buf,sizeof(uchar),size,fh) < size) {
    FatalSysError("Error writing to file '%s'",filename);
  }
  fclose(fh);

  return 1;
}

void save_map(uchar *buff, uint width, uint height, float max_val, const char *filename){
  pixel *image = mallocz<pixel>(height*width*RGB*sizeof(char));
  for (uint c = 0; c < width*height; c++){
    uchar val = clampc(ROUND((buff[c]/max_val)*255));
    image[3*c] = val;
    image[3*c+1] = val;
    image[3*c+2] = val;
  }
  write_image_to_file(image,width,height,filename);
  free(image);
}

void save_grayscale(uchar *buff, uint width, uint height, const char *filename){
  pixel *image = mallocz<pixel>(height*width*RGB*sizeof(char));
  for (uint c = 0; c < width*height; c++){
    uchar val = buff[c];
    image[3*c] = val;
    image[3*c+1] = val;
    image[3*c+2] = val;
  }
  write_image_to_file(image,width,height,filename);
  free(image);
}

void save_mask(uchar *buff, uint width, uint height){
  char buf[512];
  strcpy(buf,"img/");
  strcat(buf,settings->image_name);
  buf[strlen(buf)-4] = '\0';
  if (settings->ahd_mask == 1) {
    strcat(buf,"_mask1.ppm");
  } else if (settings->ahd_mask == 2) {
    strcat(buf,"_mask2.ppm");
  } else {
    FatalError("Unknown mask number");
  }
  pixel *image = mallocz<pixel>(height*width*RGB*sizeof(char));
  for (uint c = 0; c < width*height; c++){
    uchar val = buff[c];
    image[3*c] = val;
    image[3*c+1] = val;
    image[3*c+2] = val;
  }
  write_image_to_file(image,width,height,buf);
  free(image);
}

void free_image(img *image){
    free(image->buffer);
    free(image);
}

pixel *pad_image(pixel *src, uint width, uint height, int border) {
  int dwidth = width + border*2;
  int dheight = height + border*2;
  size_t dsize = dwidth * dheight * sizeof(pixel);
  pixel *dest = mallocz<pixel>(dsize);

  for (int y = 0; y < border; y++) {
    for (int x = 0; x < width; x++) {
      int fc = get_filter_color(x,y);
      int sx = x; /* source x zero to width */
      int sy = border-y; /* source y border to -1 */
      int dx = border+x; /* border to dwidth */
      int dy = y; /* 0 to border */
      // top border
      *get_array(dest, dx, dy, dwidth) = get_pix(src, sx, sy, width)[fc];
      // bottom border
      sy = height-2-y;
      dy = border+height+y;
      *get_array(dest, dx, dy, dwidth) = get_pix(src, sx, sy, width)[fc];
    }
  }
  // copy in image
  for (uint y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int fc = get_filter_color(x,y);
      int dx = border +x;
      int dy = border +y;
      *get_array(dest, dx, dy, dwidth) = get_pix(src, x, y, width)[fc];
    }
  }
  // left and right borders
  for (uint y = 0; y < height+border*2; y++) {
    for (int x = 0; x < border; x++){
      int fc = get_filter_color(x,y);
      int dx = border-x-1;
      *get_array(dest, dx, y, dwidth) = *get_array(dest, border+x+1, y, dwidth);
      dx = dwidth-border+x;
      *get_array(dest, dx, y, dwidth) = *get_array(dest, border+width-2-x, y, dwidth);
    }
  }
  return dest;
}

//Gray = (11*R + 16*G + 5*B) /32

void img_to_grayscale(img *image, pixel *grayscale){
  Info("Converting image to grayscale");
  for (uint y = 0; y < image->height; y++) {
    for (uint x = 0; x < image->width; x++) {
      pixel *pix = get_pixel(image,x,y);
      pixel *res = get_gray(grayscale,x,y,image->width);
      *res = (pix[R] * R_LUM +
              pix[G] * G_LUM +
              pix[B] * B_LUM)/LUM_SUM;
    }
  }
}
