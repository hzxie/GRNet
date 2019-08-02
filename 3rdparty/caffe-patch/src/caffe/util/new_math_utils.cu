// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/new_math_utils.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm_ex<double>(const CBLAS_TRANSPOSE TransA,
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
  // Note that cublas follows fortran order.
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template <>
void caffe_gpu_gemm_ex<float>(const CBLAS_TRANSPOSE TransA,
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
  // Note that cublas follows fortran order.
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, ldc));
}

}  // namespace caffe
