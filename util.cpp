/*
 * util.c
 *
 *  Created on: 28 Feb 2010
 *      Author: sjf
 */

#include <sys/times.h>
#include <unistd.h>
#include <execinfo.h>
#include "util.h"


//void show_times(img *image, struct tms *start, struct tms *end, clock_t real_time) {
//    long int ticks_per_sec = sysconf(_SC_CLK_TCK);
//
//    clock_t user_time = end->tms_utime - start->tms_utime;
//    clock_t sys_time = end->tms_stime - start->tms_stime;
//    //DebugI(user_time);
//    //DebugI(sys_time);
//    //DebugI(real_time);
//    //DebugI(ticks_per_sec);
//    //long int user_time_ms = user_time / C
//
//    double rt = (real_time * 1000) / (double)ticks_per_sec;
//    double ut = (user_time * 1000) / (double)ticks_per_sec;
//    double st = (sys_time * 1000) / (double)ticks_per_sec;
//
//    if (settings->run_tests) {
//      printf("%s, %d, %d, %f\n",settings->image_name,image->width, image->height,rt);
//    } else {
//      Info("Processing time: %s %f (ms) (%f user, %f system)",settings->image_name, rt, ut, st);
//    }
//}

void quick_show_time(struct tms *start, struct tms *end, clock_t real_time,
    const char *timer_name, int width, int height) {
  long int ticks_per_sec = sysconf(_SC_CLK_TCK);

  clock_t user_time = end->tms_utime - start->tms_utime;
  clock_t sys_time = end->tms_stime - start->tms_stime;
  //DebugI(user_time);
  //DebugI(sys_time);
  //DebugI(real_time);
  //DebugI(ticks_per_sec);
  //long int user_time_ms = user_time / C

  double rt = (real_time * 1000) / (double)ticks_per_sec;
  double ut = (user_time * 1000) / (double)ticks_per_sec;
  double st = (sys_time * 1000) / (double)ticks_per_sec;

  uint num_pixels = width * height;
  if (settings->run_tests) {
    printf("%s, %s, %d, %d, %d, %f, %f, %f, %f\n",
        timer_name, settings->image_name, width, height, num_pixels, rt, ut, st, settings->mask_size);
  } else {
    const char *unit = "ms";
    if (rt > 1000) {
      rt /= 1000;
      ut /= 1000;
      st /= 1000;
      unit = "s";
    }
    Info("Timer %s: %s: %0.3f (%s) (%0.3f user, %0.3f system)",
        settings->image_name, timer_name, rt, unit, ut, st);
  }
}

/* Obtain a backtrace and print it to stdout. */
/* libc Manual 33.1 Backtraces */
/* Compile with -rdynamic for better symbols */
/* http://www.gnu.org/software/libc/manual/html_node/Backtraces.html */
void print_stack_trace (void)
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  //printf ("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
    printf ("%s\n", strings[i]);

  free (strings);
}

