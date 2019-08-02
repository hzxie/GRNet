// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef CAFFE_NEW_MATH_UTILS_H_
#define CAFFE_NEW_MATH_UTILS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_gemm_ex(const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB,
                       const int M,
                       const int N,
                       const int K,
                       const Dtype alpha,
                       const Dtype* A,
                       const int lda,
                       const Dtype* B,
                       const int ldb,
                       const Dtype beta,
                       Dtype* C,
                       const int ldc);

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void caffe_gpu_gemm_ex(const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB,
                       const int M,
                       const int N,
                       const int K,
                       const Dtype alpha,
                       const Dtype* A,
                       const int lda,
                       const Dtype* B,
                       const int ldb,
                       const Dtype beta,
                       Dtype* C,
                       const int ldc);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_NEW_MATH_UTILS_H_
