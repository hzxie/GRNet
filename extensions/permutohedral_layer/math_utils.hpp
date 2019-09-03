/*
 * @Author: Haozhe Xie
 * @Date:   2019-08-30 10:01:53
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-09-03 16:30:03
 * @Email:  cshzxie@gmail.com
 */

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <cblas.h>
#include <cublas_v2.h>

inline void cpu_scal(const int N, const float alpha, float* X) {
  cblas_sscal(N, alpha, X, 1);
}

inline void cpu_mul(const int N, const float* a, const float* b, float* y) {
  // TODO(Haozhe Xie): accelerate with MKL
  for (int i = 0; i < N; ++i) {
    y[i] = a[i] * b[i];
  }
}

inline void cpu_mul_inverse(const int N,
                            const float* x,
                            float* y,
                            float eps = 1e-20) {
  // TODO(Haozhe Xie): accelerate with MKL
  for (int i = 0; i < N; ++i) {
    y[i] = 1.0 / (x[i] + eps);
  }
}

inline void cpu_gemm(const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB,
                     const int M,
                     const int N,
                     const int K,
                     const float alpha,
                     const float* A,
                     const float* B,
                     const float beta,
                     float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

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

void gpu_scal(const cublasHandle_t& handle,
              const int N,
              const float alpha,
              float* X);

void gpu_mul(const int N, const float* a, const float* b, float* y);

void gpu_mul_inverse(const int N, const float* x, float* y, float eps = 1e-20);

void gpu_gemm(const cublasHandle_t& handle,
              const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB,
              const int M,
              const int N,
              const int K,
              const float alpha,
              const float* A,
              const float* B,
              const float beta,
              float* C);

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

void gpu_gemv(const cublasHandle_t& handle,
              const CBLAS_TRANSPOSE TransA,
              const int M,
              const int N,
              const float alpha,
              const float* A,
              const float* x,
              const float beta,
              float* y);

#endif /* MATH_UTILS_HPP */