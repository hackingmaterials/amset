#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "amset.h"
#include "amset_array.h"

static PyObject *py_interpolate(PyObject *self, PyObject *args);
static Larray *convert_to_larray(PyArrayObject *npyary);
static Darray *convert_to_darray(PyArrayObject *npyary);

static PyMethodDef _amset_methods[] = {
    {"interpolate", (PyCFunction)py_interpolate, METH_VARARGS,
     "Interpolate wavefunction coefficients"},
    {"omp_max_threads", py_get_omp_max_threads, METH_VARARGS,
     "Return openmp max number of threads. Return 0 unless openmp is "
     "activated."},
    {NULL, NULL, 0, NULL}};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state *)PyModule_GetState(m))

static PyObject *error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static int _amset_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _amset_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "_amset",
                                       NULL,
                                       sizeof(struct module_state),
                                       _amset_methods,
                                       NULL,
                                       _amset_traverse,
                                       _amset_clear,
                                       NULL};

#define INITERROR return NULL

PyObject *PyInit__amset(void) {
    PyObject *module = PyModule_Create(&moduledef);
    struct module_state *st;

    if (module == NULL) INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("_amset.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    return module;
}

static PyObject *py_interpolate(PyObject *self, PyObject *args) {
    PyArrayObject *py_grid;
    PyArrayObject *py_data;
    PyArrayObject *py_points;

    Darray *grid;
    Darray *data;
    Darray *points;

    double *a_0, *a_1, *a_2, *a_3;
    double *b_0, *b_1, *b_2, *b_3;
    long *n_0, *n_1, *n_2, *n_3;
    double val

    if (!PyArg_ParseTuple(args, "((ddi)(ddi)(ddi)(ddi))00", &a_0, &b_0, &n_0,
                          &a_1, &b_1, &n_1, &a_2, &b_2, &n_2, &a_3, &b_3, &n_3,
                          &py_data, &py_points)) {
        return NULL;
    }

    data = convert_to_darray(py_data);
    points = convert_to_darray(py_points);

    val = amset_interpolate(a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, n_0, n_1, n_2, n_3, data, points)

    free(data);
    data = NULL;
    free(points);
    points = NULL;

    return PyDouble_FromDouble(val);
}

static PyObject *py_get_omp_max_threads(PyObject *self, PyObject *args) {
    return PyLong_FromLong(amset_get_max_threads());
}

/**
 * @brief Convert numpy "int_" array to phono3py long array structure.
 *
 * @param npyary
 * @return Larray*
 */
static Larray *convert_to_larray(PyArrayObject *npyary) {
    long i;
    Larray *ary;

    ary = (Larray *)malloc(sizeof(Larray));
    for (i = 0; i < PyArray_NDIM(npyary); i++) {
        ary->dims[i] = PyArray_DIMS(npyary)[i];
    }
    ary->data = (long *)PyArray_DATA(npyary);
    return ary;
}

/**
 * @brief Convert numpy "double" array to phono3py double array structure.
 *
 * @param npyary
 * @return Darray*
 * @note PyArray_NDIM receives non-const (PyArrayObject *).
 */
static Darray *convert_to_darray(PyArrayObject *npyary) {
    int i;
    Darray *ary;

    ary = (Darray *)malloc(sizeof(Darray));
    for (i = 0; i < PyArray_NDIM(npyary); i++) {
        ary->dims[i] = PyArray_DIMS(npyary)[i];
    }
    ary->data = (double *)PyArray_DATA(npyary);
    return ary;