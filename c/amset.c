#include "amset.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

double amset_interpolate(
    const double a_0, const double a_1, const double a_2, const double a_3,
    const double b_0, const double b_1, const double b_2, const double b_3,
    const long n_0, const long n_1, const long n_2, const long n_3,
    const Darray *C, const Darray *points) {

    double d_0, d_1, d_2, d_3;
    double u_0, u_1, u_2, u_3;
    long i_0, i_1, i_2, i_3;
    double x_0, x_1, x_2, x_3;
    double l_0, l_1, l_2, l_3;
    double p_0_0, p_0_1, p_1_0, p_1_1, p_2_0, p_2_1, p_3_0, p_3_1;
    double val;
    long num_0, num_1, num_2, num_3;

    // dim 0: uniform
    d_0 = (n_0 - 1.0) / (b_0 - a_0);
    // dim 1: uniform
    d_1 = (n_1 - 1.0) / (b_1 - a_1);
    // dim 2: uniform
    d_2 = (n_2 - 1.0) / (b_2 - a_2);
    // dim 3: uniform
    d_3 = (n_3 - 1.0) / (b_3 - a_3);

    // extract coordinates
    x_0 = points->data[0];
    x_1 = points->data[1];
    x_2 = points->data[2];
    x_3 = points->data[3];

    // compute indices and barycentric coordinates
    // dimension 0: uniform grid
    u_0 = (x_0 - a_0) * d_0;
    i_0 = floor(u_0);
    i_0 = fmax(fmin(i_0, n_0 - 2), 0);
    l_0 = u_0 - i_0;
    // dimension 1: uniform grid
    u_1 = (x_1 - a_1) * d_1;
    i_1 = floor(u_1);
    i_1 = fmax(fmin(i_1, n_1 - 2), 0);
    l_1 = u_1 - i_1;
    // dimension 2: uniform grid
    u_2 = (x_2 - a_2) * d_2;
    i_2 = floor(u_2);
    i_2 = fmax(fmin(i_2, n_2 - 2), 0);
    l_2 = u_2 - i_2;
    // dimension 3: uniform grid
    u_3 = (x_3 - a_3) * d_3;
    i_3 = floor(u_3);
    i_3 = fmax(fmin(i_3, n_3 - 2), 0);
    l_3 = u_3 - i_3;

    // Basis functions
    p_0_0 = 1.0 - l_0;
    p_0_1 = l_0;
    p_1_0 = 1.0 - l_1;
    p_1_1 = l_1;
    p_2_0 = 1.0 - l_2;
    p_2_1 = l_2;
    p_3_0 = 1.0 - l_3;
    p_3_1 = l_3;

    num_0 = C->dims[0];
    num_1 = C->dims[1];
    num_2 = C->dims[2];
    num_3 = C->dims[3];

    val =
        p_0_0 *
            (p_1_0 *
                 (p_2_0 * (p_3_0 * (C->data[i_0 * num_0 + i_1 * num_1 + i_2 * num_2 + i_3 * num_3]) +
                           p_3_1 * (C->data[i_0 * num_0 + i_1 * num_1 + i_2 * num_2 + (i_3 + 1) * num_3])) +
                  p_2_1 * (p_3_0 * (C->data[i_0 * num_0 + i_1 * num_1 + (i_2 + 1) * num_2 + i_3 * num_3]) +
                           p_3_1 * (C->data[i_0 * num_0 + i_1 * num_1 + (i_2 + 1) * num_2 + (i_3 + 1) * num_3]))) +
             p_1_1 *
                 (p_2_0 * (p_3_0 * (C->data[i_0 * num_0 + (i_1 + 1) * num_1 + i_2 * num_2 + i_3 * num_3]) +
                           p_3_1 * (C->data[i_0 * num_0 + (i_1 + 1) * num_1 + i_2 + (i_3 + 1) * num_3])) +
                  p_2_1 *
                      (p_3_0 * (C->data[i_0 * num_0 + (i_1 + 1) * num_1 + (i_2 + 1) * num_2 + i_3 * num_3]) +
                       p_3_1 * (C->data[i_0 * num_0 + (i_1 + 1) * num_1 + (i_2 + 1) * num_2 + (i_3 + 1) * num_3])))) +
        p_0_1 *
            (p_1_0 *
                 (p_2_0 * (p_3_0 * (C->data[(i_0 + 1) * num_0 + i_1 * num_1 + i_2 * num_2 + i_3 * num_3]) +
                           p_3_1 * (C->data[(i_0 + 1) * num_0 + i_1 * num_1 + i_2 * num_2 + (i_3 + 1) * num_3])) +
                  p_2_1 *
                      (p_3_0 * (C->data[(i_0 + 1) * num_0 + i_1 * num_1 + (i_2 + 1) * num_2 + i_3 * num_3]) +
                       p_3_1 * (C->data[(i_0 + 1) * num_0 + i_1 * num_1 + (i_2 + 1) * num_2 + (i_3 + 1) * num_3]))) +
             p_1_1 *
                 (p_2_0 *
                      (p_3_0 * (C->data[(i_0 + 1) * num_0 + (i_1 + 1) * num_1 + i_2 * num_2 + i_3 * num_3]) +
                       p_3_1 * (C->data[(i_0 + 1) * num_0 + (i_1 + 1) * num_1 + i_2 * num_2 + (i_3 + 1) * num_3])) +
                  p_2_1 *
                      (p_3_0 * (C->data[(i_0 + 1) * num_0 + (i_1 + 1) * num_1 + (i_2 + 1) * num_2 + i_3 * num_3]) +
                       p_3_1 *
                           (C->data[(i_0 + 1) * num_0 + (i_1 + 1) * num_1 + (i_2 + 1) * num_2 + (i_3 + 1) * num_3]))));

    return val;
}

long amset_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}