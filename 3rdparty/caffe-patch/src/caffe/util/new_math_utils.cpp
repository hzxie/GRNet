// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include "caffe/util/new_math_utils.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_cpu_gemm_ex<float>(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB,
                              const int M,
                              const int N,
                              const int K,
                              const float alpha,
                              const float* A,
                              const int lda,
                              const float* B,
                              const int ldb,
                              const float beta,
                              float* C,
                              const int ldc) {
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void caffe_cpu_gemm_ex<double>(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB,
                               const int M,
                               const int N,
                               const int K,
                               const double alpha,
                               const double* A,
                               const int lda,
                               const double* B,
                               const int ldb,
                               const double beta,
                               double* C,
                               const int ldc) {
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

}  // namespace caffe
