#!python
# cython: embedsignature=True
# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Oct 16, 2013

Fast cython functions for calculating various filters

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import platform

cimport numpy as np
import numpy as np
from libc.math cimport exp, fabs
cimport cython


DTYPE_float = np.float

ctypedef np.float_t DTYPE_f
ctypedef np.double_t DTYPE_d
ctypedef np.int_t DTYPE_i

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def exp_filter(np.ndarray[DTYPE_d, ndim=1] in_data,
               np.ndarray[DTYPE_d, ndim=1] in_jd,
               int ctime=10,
               double nan=-999999.0):
    """
    Calculates exponentially smoothed time series using an
    iterative algorithm

    Parameters
    ----------
    in_data : double numpy.array
        input data
    in_jd : double numpy.array
        julian dates of input data
    ctime : int
        characteristic time used for calculating
        the weight
    nan : double
        nan values to exclude from calculation
    """
    cdef np.ndarray[DTYPE_f, ndim = 1] filtered = np.empty(len(in_data))
    cdef double tdiff
    cdef float ef
    cdef float nom = 1
    cdef float denom = 1
    cdef bint isnan
    cdef double last_jd_var
    cdef unsigned int i

    filtered.fill(np.nan)

    last_jd_var = in_jd[0]

    for i in range(in_jd.shape[0]):
        isnan = (in_data[i] == nan) or npy_isnan(in_data[i])
        if not isnan:
            tdiff = in_jd[i] - last_jd_var
            ef = exp(-tdiff / ctime)
            nom = ef * nom + in_data[i]
            denom = ef * denom + 1
            last_jd_var = in_jd[i]
            filtered[i] = nom / denom

    return filtered


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def boxcar_filter(np.ndarray[DTYPE_d, ndim=1] in_data,
                  np.ndarray[DTYPE_d, ndim=1] in_jd,
                  float window=1, double nan=-999999.0):
    """
    Calculates filtered time series using
    a boxcar filter - basically a moving average calculation

    Parameters
    ----------
    in_data : double numpy.array
        input data
    in_jd : double numpy.array
        julian dates of input data
    window : int
        characteristic time used for calculating
        the weight
    nan : double
        nan values to exclude from calculation
    """
    cdef np.ndarray[DTYPE_f, ndim = 1] filtered = np.empty(len(in_data))
    cdef double tdiff
    cdef unsigned int i
    cdef unsigned int j
    cdef bint isnan
    cdef double sum = 0
    cdef double limit = window / 2.0
    cdef int nobs = 0

    filtered.fill(np.nan)

    for i in range(in_jd.shape[0]):
        isnan = (in_data[i] == nan) or npy_isnan(in_data[i])
        if not isnan:
            sum = 0
            nobs = 0
            for j in range(i, in_jd.shape[0]):
                isnan = (in_data[j] == nan) or npy_isnan(in_data[j])
                if not isnan:
                    tdiff = in_jd[j] - in_jd[i]
                    if fabs(tdiff) <= limit:
                        sum = sum + in_data[j]
                        nobs = nobs + 1
                    else:
                        break

            for j in range(i, -1, -1):
                if i != j:
                    isnan = (in_data[j] == nan) or npy_isnan(in_data[j])
                    if not isnan:
                        tdiff = in_jd[j] - in_jd[i]
                        if fabs(tdiff) <= limit:
                            sum = sum + in_data[j]
                            nobs = nobs + 1
                        else:
                            break

            filtered[i] = sum / nobs

    return filtered
