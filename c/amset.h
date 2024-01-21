#ifndef __phono3py_H__
#define __phono3py_H__

#include "amset_array.h"

double amset_interpolate(
    const double a_0, const double a_1, const double a_2, const double a_3,
    const double b_0, const double b_1, const double b_2, const double b_3,
    const long n_0, const long n_1, const long n_2, const long n_3,
    const Darray *data, const Darray *points);

long amset_get_max_threads(void);