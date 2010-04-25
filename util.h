/*
 * util.h
 *
 *  Created on: 28 Feb 2010
 *      Author: sjf
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "image.h"
void show_times(img* image, struct tms *start, struct tms *end, clock_t real_time, settings_t *settings);
void quick_show_time(struct tms *start, struct tms *end, clock_t real_time, const char *s,int width, int height);
void print_stack_trace (void);
#endif /* UTIL_H_ */
