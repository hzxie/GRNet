// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include "math_utils.hpp"

inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition)                                        \
  do {                                                                 \
    cublasStatus_t status = condition;                                 \
    if (status != CUBLAS_STATUS_SUCCESS) cublasGetErrorString(status); \
  } while (0)

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
                 const int ldc) {
  // Note that cublas follows fortran order.
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, ldc));
}
