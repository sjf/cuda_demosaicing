/*
 * base.h
 *
 *  Created on: 28 Jan 2010
 *      Author: sjf65
 */

#ifndef BASE_H_
#define BASE_H_

#include <sys/types.h>
#include <sys/times.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

typedef unsigned char uchar;
typedef short int cbool;

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif
#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif
#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
#define SWAP(a,b) {void *swap; swap = a; a = b; b = (typeof(b))swap;}
#define SQ(x) ((x)*(x))
#define ABS(x) (((x) < 0) ? (-(x)) : (x))

#define MB(n)    ((n)/1024.0/1024.0)

/* Logging Macros */

#ifdef _DEBUG_
#define Debug(x, args...) printf(x, ## args)
#define DebugI(x) do {printf("%s: %d\n",#x,x);} while(0)
#define DebugF(x) do {printf("%s: %f\n",#x,x);} while(0)
#define DebugX(x) do {printf("%s: 0x%x\n",#x,x);} while(0)
#define DebugS(x) do {printf("%s: '%s'\n",#x,x);} while(0)
#else
#define Debug(x, args...) do{}while(0)
#define DebugMAC(x) do{}while(0)
#define DebugI(x) do {} while(0)
#define DebugS(x) do {} while(0)
#endif

// When compiled with debug, dump core
// Normal execution, exit with code 1
#ifdef _DEBUG_
#define fail() abort()
#else
#define fail() exit(1)
#endif

#define FatalError(x, args...) do{fprintf(stderr," !! error at %s:%i:%s ", __FILE__,__LINE__,__FUNCTION__); \
                                  fprintf(stderr,x, ##args); fprintf(stderr,"\n"); \
                                  fail(); \
                               } while(0)

#define FatalSysError(x, args...) do{perror(" !! error:"); \
                                  fprintf(stderr," !! error at %s:%i:%s ", __FILE__,__LINE__,__FUNCTION__); \
                                  fprintf(stderr,x, ##args); fprintf(stderr,"\n"); \
                                  fail(); \
                               } while(0)

#define Error(x, args...) do{fprintf(stderr," ## error at %s:%i:%s ", __FILE__,__LINE__,__FUNCTION__); \
                             fprintf(stderr,x, ##args); fprintf(stderr,"\n"); \
                              } while(0)
#define Warn(x, args...) do{printf(" == warning at %s:%i:%s ", __FILE__,__LINE__,__FUNCTION__); \
                             printf(x, ##args); printf("\n"); \
                          } while(0)
#define Info(x, args...) do {printf(" ++ ");printf(x, ## args);printf("\n"); \
                                                  } while (0)
/*
#define InfoFile(x, args...) do {printf(" ++ ");printf(x,##args);printf("\n");\
                                 fprintf(sr->log,"%lu ",time(NULL));fprintf(sr->log,x, ## args);\
                                 fprintf(sr->log,"\n");fflush(sr->log); } while (0)
*/

#define Todo(x, args...) do {printf(" ** Todo at %s: ",__FUNCTION__);printf(x, ## args);printf("\n"); \
                          } while (0)
#define Fine(x, args...) do {printf("    ");printf(x, ## args);printf("\n"); \
                          } while (0)

#define startTimer(n)  struct tms start##n; clock_t starttime##n = times(&start##n); \
                        Info("Starting timer %s", #n);
#define stopTimer(n,w,h)   struct tms   end##n; clock_t   endtime##n = times(&end##n); \
                           quick_show_time(&start##n, &end##n, \
                                         endtime##n - starttime##n, #n,(w),(h));
#define quickStop(n)   struct tms   end##n; clock_t   endtime##n = times(&end##n); \
                           quick_show_time(&start##n, &end##n, \
                                         endtime##n - starttime##n, #n,0,0);
#define CV_CIE 0
#define EZ_RGB_TO_CIE 1
#define RGB_ONLY 2
// default edge detection threshold
#define THRESHOLD 50

/* Global Structs */
typedef struct {
  /* Driver program settings */
  uint run_tests;
  uint save_temps;
  /* Interpolation settings */
  uchar use_cuda;
  uchar use_ahd;
  uchar ahd_mask;
  /* Size of the homogenity neighbourhood. Values may be 1 or 2, (default 2) */
  uchar ball_distance;
  uchar colorspace_conversion;
  uint median_filter_iterations;
  uint dilations;
  uint edge_threshold;
  /* Image specific settings */
  char *image_name;
  int bayer_type;

  /* Percentage of the image that is covered by the mask. */
  float mask_size;
} settings_t;

extern settings_t *settings;

/* Function declarations */

char *time_str();
void init_settings(settings_t *settings);
void init_image_settings(char *image_name);
void *callocz(size_t nmemb, size_t size);
void out_filename(const char *filename, char *buf, size_t len);
int intcmp(const void *p1, const void *p2);


template <class T>
T *mallocz(size_t size) {
    void *buf = malloc(size);
#ifndef _TEST
    bzero(buf,size);
#endif
    return (T *)buf;
}

/* malloczero: malloc and bzero always */
template <class T>
T *malloczero(size_t size) {
    void *buf = malloc(size);
    bzero(buf,size);
    return (T *)buf;
}


#endif /* BASE_H_ */
