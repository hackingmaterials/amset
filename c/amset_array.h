#ifndef __amset_array_H__
#define __amset_array_H__

#define MAX_NUM_DIM 20

/* It is assumed that number of dimensions is known for each array. */
typedef struct {
    long dims[MAX_NUM_DIM];
    long *data;
} Larray;

typedef struct {
    int dims[MAX_NUM_DIM];
    double *data;
} Darray;
