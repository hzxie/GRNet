// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <cblas.h>
#include <cublas_v2.h>

inline void cpu_gemm_ex(const CBLAS_TRANSPOSE TransA,
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

void gpu_gemm_ex(const cublasHandle_t& handle,
                 const CBLAS_TRANSPOSE TransA,
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
                 const int ldc);

#endif /* MATH_UTILS_HPP */