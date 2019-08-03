// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

// atomicAdd is not defined for doubles in Cuda 7.5
// from http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  if (val == 0.0) return __longlong_as_double(old);
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif
#endif
